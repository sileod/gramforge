"""Premise references in a proof do not certify that a question needs premises."""
import gzip
import json

import pytest
from gramforge.fol_nli.curate import candidates
from gramforge.fol_nli.solver import premise_dependence


@pytest.mark.parametrize('hypothesis,label,expected', [
    ('p<~>p', 'contradiction', 'independent'),
    ('~(p<~>p)', 'entailment', 'independent'),
    ('p=>p', 'entailment', 'independent'),
    ('q', 'contradiction', 'dependent'),
    ('q', 'entailment', 'dependent'),
])
def test_dependence_checks_theory_without_premises(hypothesis, label, expected):
    assert premise_dependence([], hypothesis, label)['status'] == expected


def pool(tmp_path):
    group = {'index': 0, 'background': [], 'clauses': [{'id':'p0', 'formula':'~q&r', 'english':'q is false and r is true'}],
             'name_lexicon': {}, 'mentioned_names': [], 'lexicon': {}, 'proposition_lexicon': {},
             'rule_concentration': 0, 'outcomes': []}
    for formula in ['p<~>p', 'q']:
        group['outcomes'].append({'label':'contradiction', 'hypothesis': {'formula': formula, 'english': formula},
                                  'proof_inputs': ['p0'], 'proof_trace': ['resolution'], 'proof_check': None})
    directory = tmp_path / 'groups'
    directory.mkdir()
    with gzip.open(directory / '00000000.json.gz', 'wt') as file:
        json.dump(group, file)
    return tmp_path


def test_nonminimal_proof_cannot_admit_premise_independent_query(tmp_path):
    rows, rejected = candidates(pool(tmp_path), 24, 5000, 42)
    assert rejected['premise_independent'] == 1
    assert [r['hypothesis_tptp'] for r in rows] == ['q']
    assert rows[0]['dependency_check']['status'] == 'dependent'


def test_unknown_dependency_never_passes_curation(tmp_path, monkeypatch):
    monkeypatch.setattr('gramforge.fol_nli.curate.premise_dependence',
                        lambda *args: {'status':'unresolved'})
    rows, rejected = candidates(pool(tmp_path), 24, 5000, 42)
    assert rows == []
    assert rejected['premise_unresolved'] == 2


def test_opposite_query_cannot_reintroduce_whole_premise_copy(tmp_path):
    root = pool(tmp_path)
    path = root / 'groups/00000000.json.gz'
    with gzip.open(path, 'rt') as file:
        group = json.load(file)
    group['clauses'][0].update(formula='~q', english='q is false')
    for formula, label in [('~(q)', 'entailment'), ('~(~(~q))', 'entailment')]:
        group['outcomes'].append({'label':label, 'hypothesis':{'formula':formula, 'english':formula},
                                  'proof_inputs':['p0'], 'proof_trace':['resolution'], 'proof_check':None})
    with gzip.open(path, 'wt') as file:
        json.dump(group, file)
    rows, rejected = candidates(root, 24, 5000, 42)
    assert rows == []
    assert rejected['direct_premise_match'] == 3  # q and both equivalent ~q queries.
    assert rejected['premise_independent'] == 1


def test_setup_definitions_and_clause_count_do_not_reveal_proof_usage():
    from gramforge.fol_nli.curate import select_clauses
    clauses = [{'id':f'p{i}', 'formula':f'q{i}', 'english':f'q{i}'} for i in range(12)]
    clauses[9]['role'] = 'definition'
    for support in [set(), {'p3'}, {'p0','p4','p9'}]:
        selected = select_clauses(clauses, support, 6, str(support))
        identifiers = {c['id'] for c in selected}
        assert {'p0','p9'} | support <= identifiers
        assert len(selected) == 6
    assert select_clauses(clauses, {'p1','p2'}, 3, 'overflow') is None


def test_room_setup_omission_is_decided_for_all_queries_before_labels():
    from gramforge.fol_nli.generate import prepare_surface
    setup = {'id':'p0', 'formula':'room(mary)', 'english':'Mary is in the room.'}
    fact = {'id':'p1', 'formula':'p', 'english':'p'}
    queries = [{'formula':'q','english':'q'}, {'formula':'room(mary)','english':'Mary is in the room'}]
    retained, _, _ = prepare_surface([setup,fact], queries, 42, 0)
    assert {c['id'] for c in retained} == {'p0','p1'}
    retained, _, _ = prepare_surface([setup,fact], queries[:1], 42, 0)
    assert [c['id'] for c in retained] == ['p1']


def test_group_surface_paraphrases_preserve_room_scope_and_formulas():
    from gramforge.fol_nli.generate import prepare_surface
    clauses = [{'id':'p0','formula':'room(mary)','english':'Mary is the only person in the room.'},
               {'id':'p1','formula':'![X]:(room(X)=>p(X))','english':'everyone in the room smiles'}]
    queries = [{'formula':'?[X]:(room(X)&p(X))','english':'someone in the room smiles'}]
    styles = {}
    for index in range(10):
        rendered, questions, style = prepare_surface(clauses,queries,42,index)
        styles[style] = (rendered, questions)
        assert [c['formula'] for c in rendered] == [c['formula'] for c in clauses]
        assert [h['formula'] for h in questions] == [h['formula'] for h in queries]
    assert set(styles) == {'room','occupants'}
    assert styles['occupants'][0][1]['english'] == 'every room occupant smiles'
    assert styles['occupants'][1][0]['english'] == 'a room occupant smiles'


@pytest.mark.parametrize('size', [6,7,8])
def test_surface_matching_prevents_definition_free_neutral_shortcut(size):
    from collections import Counter
    from gramforge.fol_nli.curate import LABELS, match_surface
    from gramforge.fol_nli.surface import stratum
    def row(label, i, defined):
        return {'id':f'{label}-{i}', 'label':label, 'hypothesis':'q',
                'premise':('Definition: ' + 'long ' * 130) if defined else 'short'}
    pools = {label:[row(label,0,True), row(label,1,True), row(label,2,False)] for label in LABELS}
    pools['neutral'] = [row('neutral',i,False) for i in range(3,33)] + pools['neutral']
    selected = match_surface(pools,size)
    assert len(selected) == size
    for key in {stratum(r) for r in selected}:
        counts = Counter(r['label'] for r in selected if stratum(r) == key)
        assert max(counts.get(label,0) for label in LABELS) - min(counts.get(label,0) for label in LABELS) <= 1
    if size == 6:
        assert sum(r['label']=='neutral' and 'Definition:' in r['premise'] for r in selected) == 1


def test_missing_surface_matches_fail_instead_of_relaxing_balance():
    from gramforge.fol_nli.curate import match_surface
    row = lambda label, text: {'id':label,'label':label,'premise':text,'hypothesis':'q'}
    pools = {'entailment':[row('entailment','Definition: p')],
             'contradiction':[row('contradiction','Definition: p')],
             'neutral':[row('neutral','short')]}
    with pytest.raises(ValueError, match='matched surface'):
        match_surface(pools,3)


def test_paired_clause_verification_accepts_arrow_nulls_but_rejects_text_changes():
    import pyarrow as pa
    from gramforge.fol_nli.verify import record_value
    clauses = [{'id':'p0','formula':'p','english':'p','derivation':None,'role':'setup'},
               {'id':'p1','formula':'q','english':'q','derivation':{'rule':1,'state':{'k':1}}}]
    decoded = pa.array([clauses]).to_pylist()[0]
    assert decoded != clauses  # Arrow adds role=None to the second clause.
    assert record_value(decoded) == record_value(clauses)
    decoded[1]['english'] = 'not q'
    assert record_value(decoded) != record_value(clauses)
