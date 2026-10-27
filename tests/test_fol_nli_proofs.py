"""Portfolio proofs must survive supplementary proof-search failures."""
import pytest
from gramforge.fol_nli import solver

PROOF = "fof(1,axiom,p,file('problem.p',p1)).\nfof(2,plain,$false,inference(resolution,[],[1]))."


@pytest.mark.parametrize('verdict,statuses', [
    ('entailment', ['Satisfiable', 'Unsatisfiable']),
    ('contradiction', ['Unsatisfiable', 'Satisfiable']),
])
def test_usable_portfolio_proof_never_reruns(monkeypatch, verdict, statuses):
    calls = []
    def check(*args, **kwargs):
        calls.append((args, kwargs))
        status = statuses[len(calls) - 1]
        return {'status': status, 'output': PROOF if status == 'Unsatisfiable' else ''}
    monkeypatch.setattr(solver, 'check', check)
    result = solver.label([], [{'id': 'p1', 'formula': 'p'}], 'p', 'vampire')
    assert len(calls) == 2
    assert result['label'] == verdict
    assert result['proof_inputs'] == ['p1']
    assert result['proof'] == PROOF
    assert result['proof_source'] == 'portfolio'
    assert result['proof_check'] is None


@pytest.mark.parametrize('success', [False, True])
def test_missing_support_gets_larger_supplement_without_changing_label(monkeypatch, success):
    calls = []
    portfolio = '% SZS status Unsatisfiable\n% proof without named inputs'
    answers = [{'status': 'Satisfiable', 'output': ''},
               {'status': 'Unsatisfiable', 'output': portfolio},
               {'status': 'Unsatisfiable' if success else 'Timeout', 'output': PROOF if success else ''}]
    def check(*args, **kwargs):
        calls.append((args, kwargs))
        return answers[len(calls) - 1]
    monkeypatch.setattr(solver, 'check', check)
    result = solver.label([], [{'id': 'p1', 'formula': 'p'}], 'p', 'vampire', seconds=2)
    assert result['label'] == 'entailment'
    assert calls[2][0][2] == 10 and calls[2][1]['mode'] == 'vampire'
    assert result['proof'] == (PROOF if success else portfolio)
    assert result['proof_support_missing'] is not success
    assert result['proof_source'] == ('supplement' if success else 'portfolio')
