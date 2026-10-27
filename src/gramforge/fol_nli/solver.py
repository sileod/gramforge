"""Vampire checks: unknown/timeouts are never labels. No shell commands."""
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path


def problem(background, clauses, hypothesis=None, negate=False):
    lines = [f'fof(b{i},axiom,({formula})).' for i, formula in enumerate(background)]
    lines += [f'fof({c["id"]},axiom,({c["formula"]})).' for c in clauses]
    if hypothesis is not None:
        formula = f'~({hypothesis})' if negate else hypothesis
        lines.append(f'fof(query,axiom,({formula})).')
    return '\n'.join(lines) + '\n'


def check(text, executable, seconds=2, mode="casc_sat"):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.p') as file:
        file.write(text); file.flush()
        try:
            result = subprocess.run([str(executable), file.name, '--mode', mode,
                                     '--output_axiom_names', 'on', '--proof', 'tptp', '-t', f'{seconds}s'],
                                    capture_output=True, text=True, timeout=seconds + 5)
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            output = '% subprocess timeout'
    match = re.search(r'SZS status (\w+)', output)
    return {'status': match[1] if match else 'Unresolved', 'output': output,
            'problem_sha256': hashlib.sha256(text.encode()).hexdigest()}


def label(background, clauses, hypothesis, executable, seconds=2):
    positive = check(problem(background, clauses, hypothesis), executable, seconds)
    negative = check(problem(background, clauses, hypothesis, negate=True), executable, seconds)
    statuses = positive['status'], negative['status']
    labels = {('Satisfiable', 'Unsatisfiable'): 'entailment',
              ('Unsatisfiable', 'Satisfiable'): 'contradiction',
              ('Satisfiable', 'Satisfiable'): 'neutral',
              ('Unsatisfiable', 'Unsatisfiable'): 'inconsistent'}
    verdict = labels.get(statuses, 'unresolved')
    proof = negative['output'] if verdict == 'entailment' else positive['output'] if verdict == 'contradiction' else ''
    proof_check = None
    if verdict in ('entailment', 'contradiction'):
        proof_check = check(problem(background, clauses, hypothesis, negate=verdict == 'entailment'), executable, seconds, mode='vampire')
        proof = proof_check['output'] if proof_check['status'] == 'Unsatisfiable' else ''
    support = sorted(set(re.findall(r'file\([^\n]*?,\s*(p\d+)\)', proof)))
    trace = re.findall(r'inference\((\w+)', proof)
    return {'label': verdict, 'positive': positive, 'negative': negative,
            'proof_inputs': support, 'proof_trace': trace, 'proof_check': proof_check}
