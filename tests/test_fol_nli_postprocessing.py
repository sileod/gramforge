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
    group = {'index': 0, 'background': [], 'clauses': [{'id':'p0', 'formula':'~q', 'english':'q is false'}],
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
