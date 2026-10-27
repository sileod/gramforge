"""Seeded Gramforge derivations with paired, stable premise units."""
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
import re

from gramforge import generate
from .grammar import make_grammar, background, has_self_relation
from .logic import parse, symbols


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def trace(node):
    return {'rule': node.rule.index, 'signature': node.rule.signature,
            'state': dict(node.local_state), 'children': [trace(c) for c in node.children]}


def clause(node, index):
    formula, english = node @ 'tptp', node @ 'eng'
    parse(formula)
    if '#' in english or '\\?' in formula:
        raise ValueError('Incomplete rendering')
    return {'id': f'p{index}', 'formula': formula, 'english': english, 'derivation': trace(node)}


@lru_cache(maxsize=32)
def grammar(size):
    rules = make_grammar(size)
    query = rules('query(term)', '0', '0')
    original_get_rules = rules.get_rules
    rules.allowed_terminals = {}
    def get_rules(cls, name, *args, **kwargs):
        options = original_get_rules(name, *args, **kwargs)
        allowed = cls.allowed_terminals.get(name)
        return options if allowed is None else [r for r in options if r.templates['tptp'] in allowed]
    rules.get_rules = classmethod(get_rules)
    return rules, query, [r.weight for r in rules._instances]


def make_group(seed, index, hypotheses=12):
    rng = random.Random(f'gramforge-fol-nli:{seed}:{index}')
    size = rng.randint(1, 32)
    rules, query, weights = grammar(size)
    rules.allowed_terminals.clear()
    vocabulary = json.loads((Path(__file__).parent / 'vocabulary.json').read_text())
    phrases = rng.sample(vocabulary['predicates'], 26)
    propositions = rng.sample(vocabulary['short_propositions'], 5)
    name_lexicon = dict(zip(['mary','paul','fred','alice','john','susan','lucy'], rng.sample(vocabulary['names'], 7)))
    for rule in rules._instances:
        if rule.name == 'atomic_predicate':
            rule.templates['eng'] = phrases[ord(rule.templates['tptp'][-1]) - ord('a')]
        elif rule.name == 'entity_':
            rule.templates['eng'] = name_lexicon[rule.templates['tptp']]
        elif rule.name == 'prop':
            atom = rule.templates['tptp'].lstrip('~')
            positive = propositions[ord(atom[-1]) - ord('a')].rstrip('.')
            rule.templates['eng'] = f'it is not the case that “{positive}”' if rule.templates['tptp'].startswith('~') else positive
    concentration = (rng.random() * .8) ** 1.5 if rng.random() > .1 else 0.
    # Deterministic rule-family concentration; no global NumPy reseeding.
    for rule, weight in zip(rules._instances, weights):
        factor = max(.05, rng.random() ** (1 + 4 * concentration)) if rule.args and rule.name != 'block' else 1.
        rule.weight = weight * factor
    profile = 'chain-rich' if rng.random() < .25 else 'mixed'
    if profile == 'chain-rich':
        # A sampling stratum, not a restricted grammar: every production keeps
        # positive weight, and three quarters of groups retain mixed sampling.
        for rule, base_weight in zip(rules._instances, weights):
            if rule.signature == 'term(X_quantifier,X_property)':
                rule.weight *= 4
            elif rule.signature == 'term(entity,property)':
                rule.weight *= 3
            elif rule.signature == 'property(state)':
                rule.weight *= 2 if base_weight == 12 else 4
            elif rule.name == 'group' and rule.templates['tptp'] == 'anywhere':
                rule.weight *= 30
            elif rule.name == 'block' and any(arg == 'cond' for arg in rule.args):
                rule.weight *= .1
    production = generate(rules.start(), seed=rng.getrandbits(32), depth=14, n_iter=20, max_steps=10000, k=8, save_prob=.1)
    setup = production.children[0]
    block = next(node for node in production.descendants if node.type == 'block')
    if block.rule.signature == 'block(list)':
        units = [block.children[0]]
    else:
        units = list(block.children)
    clauses = [clause(node, i + 1) for i, node in enumerate(units)]
    # Read setup from its actual derivation, never by splitting English lines.
    persons = list(setup.state.get('persons', []))
    if persons:
        room = '&'.join([f'room({n})' for n in persons])
        closure = '|'.join(f'X={n}' for n in persons)
        setup_formula = f'({room})&(![X]:(room(X)=>({closure})))'
        verb, noun = ('is', 'person') if len(persons) == 1 else ('are', 'people')
        setup_english = f'{", ".join(name_lexicon[n] for n in persons)} {verb} the only {noun} in the room.'
    else:
        setup_formula, setup_english = 'there_is_a_room', 'There is a room.'
    clauses.insert(0, {'id': 'p0', 'formula': setup_formula, 'english': setup_english, 'derivation': None})
    mentioned_names = sorted(set(re.findall(r'\b(?:mary|paul|fred|alice|john|susan|lucy)\b', ' '.join(c['formula'] for c in clauses))))
    theory = background(mentioned_names)
    arities = {}
    for c in clauses:
        for name, arity in symbols(parse(c['formula'])).items():
            if name in arities and arities[name] != arity:
                raise ValueError(f'Conflicting arity: {name}')
            arities[name] = arity
    candidates = {}
    # Reuse full embedded statements (not formula fragments) to ask questions
    # about the generated conditionals, as well as new same-vocabulary queries.
    nodes = [n for n in production.descendants if n.type == 'term']
    for node in nodes:
        formula, english = node @ 'tptp', node @ 'eng'
        if any(c['formula'] == formula for c in clauses):
            continue
        candidates[formula] = {'formula': formula, 'english': english, 'derivation': trace(node)}
    # Notebook-style concept matching: ask about properties actually occurring
    # in the premise, instantiated for its named participants. This introduces
    # no premise or rule and leaves all formula families available.
    names = sorted(set(re.findall(r'\b(?:mary|paul|fred|alice|john|susan|lucy)\b', ' '.join(c['formula'] for c in clauses))))
    for node in production.descendants:
        if node.type != 'property':
            continue
        template = node @ 'tptp'
        for name in names:
            formula = template.replace('?', name)
            parse(formula)
            if any(c['formula'] == formula for c in clauses):
                continue
            candidates[formula] = {'formula': formula, 'english': f'{name_lexicon[name]} {node @ "eng"}',
                                   'derivation': {'instantiated_property': trace(node), 'name': name}}
    # Avoid spending solver work on hypotheses containing unseen predicates.
    for category in ['atomic_predicate', 'adjective', 'predicate', 'comparative', 'sentiment', 'prop', 'noun']:
        rules.allowed_terminals[category] = set(arities)
    rules.allowed_terminals['entity_'] = set(names)
    for _ in range(2 * hypotheses):
        try:
            node = generate(query, seed=rng.getrandbits(32), depth=10, n_iter=3, max_steps=1000, k=8, save_prob=.05)
        except ValueError:
            continue
        formula = node @ 'tptp'
        if not symbols(parse(formula)).keys() <= arities.keys():
            continue
        if any(c['formula'] == formula for c in clauses):
            continue
        candidates[formula] = {'formula': formula, 'english': node @ 'eng', 'derivation': trace(node)}
    choices = sorted(formula for formula in candidates if not has_self_relation(formula))
    rng.shuffle(choices)
    # Opposite polarities provide paired, meaningful contradiction candidates.
    selected = []
    for formula in choices[:max(1, hypotheses // 2)]:
        h = candidates[formula]
        selected += [h, {'formula': f'~({formula})', 'english': f'It is not the case that “{h["english"]}”',
                         'derivation': {'negated_query': h['derivation']}}]
    return {'index': index, 'seed': seed, 'group_id': digest([theory, [c['formula'] for c in clauses]]),
            'clauses': clauses, 'background': theory, 'hypotheses': selected,
            'derivation': trace(production), 'rule_concentration': concentration, 'sampling_profile': profile,
            'lexicon': dict(zip(['pred' + chr(ord('a') + i) for i in range(26)], phrases)),
            'name_lexicon': name_lexicon, 'mentioned_names': mentioned_names,
            'proposition_lexicon': dict(zip(['proposition' + chr(ord('a') + i) for i in range(5)], propositions)),
            'requested_premise_units': size, 'domain': 'open, person-only; finite room optional'}
