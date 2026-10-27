"""Regression witnesses for meanings that previously corrupted labels."""
import pytest
from gramforge.generate_sequential import FastProduction
from gramforge.fol_nli.grammar import make_grammar, background, negative
from gramforge.fol_nli.logic import parse, z3_check
from gramforge.fol_nli.solver import problem


@pytest.fixture
def grammar():
    return make_grammar(4)


def node(grammar, signature, *children, formula=None):
    rules = [r for r in grammar._instances if r.signature == signature]
    if formula is not None:
        rules = [r for r in rules if r.templates['tptp'] == formula]
    assert rules, (signature, formula)
    result = FastProduction(rule=rules[0], children=list(children))
    for child in result.children:
        child.parent = result
    return result


def prop(grammar):
    atom = node(grammar, 'atomic_predicate', formula='preda')
    return node(grammar, 'property(atomic_predicate)', atom, formula='{0}(?)')


def equivalent(actual, intended):
    assert z3_check([f'({actual})<~>({intended})']) == 'unsat'


def test_no_one_restricts_with_conjunction(grammar):
    q = node(grammar, 'quantifier', formula=r'~\?')
    group = node(grammar, 'group', formula='room')
    quantifier = node(grammar, 'X_quantifier(quantifier,group)', q, group)
    body = node(grammar, 'X_property(property)', prop(grammar))
    sentence = node(grammar, 'term(X_quantifier,X_property)', quantifier, body)
    equivalent(sentence @ 'tptp', '~?[X]:(room(X)&preda(X))')
    assert z3_check([sentence @ 'tptp', '~room(mary)', 'preda(mary)']) == 'sat'
    assert 'no one in the room' in sentence @ 'eng'


def test_exactly_one_ignores_outsiders(grammar):
    group = node(grammar, 'group', formula='room')
    rules = [r for r in grammar._instances if r.signature == 'term(property,group)'
             and r.templates['eng'] == 'only one person {1} {0}']
    assert len(rules) == 1
    sentence = FastProduction(rule=rules[0], children=[prop(grammar), group])
    equivalent(sentence @ 'tptp', '(?[X]:(room(X)&preda(X)))&(![X,Y]:((room(X)&room(Y)&preda(X)&preda(Y))=>X=Y))')
    assert z3_check([sentence @ 'tptp', 'room(mary)', 'preda(mary)', '~room(paul)', 'preda(paul)', 'mary!=paul']) == 'sat'


def test_superlative_requires_group_membership(grammar):
    name = node(grammar, 'entity_', formula='mary')
    entity = node(grammar, 'entity(entity_)', name)
    group = node(grammar, 'group', formula='room')
    rule = next(r for r in grammar._instances if r.signature == 'term(entity,group)')
    sentence = FastProduction(rule=rule, children=[entity, group])
    assert z3_check([sentence @ 'tptp', '~room(mary)']) == 'unsat'


def test_comparison_is_subject_first_and_strict():
    theory = background()
    assert z3_check([*theory, 'older(mary,paul)', 'older(paul,mary)']) == 'unsat'
    assert z3_check([*theory, 'older(mary,mary)']) == 'unsat'
    assert z3_check([*theory, 'older(mary,paul)', '~younger(paul,mary)']) == 'unsat'
    assert z3_check([*theory, 'older(mary,paul)', 'older(paul,fred)', '~older(mary,fred)']) == 'unsat'


def test_open_domain_is_not_closed_to_named_people():
    # An eighth distinct person is permitted; this is not a finite enumeration.
    theory = background() + [f'visitor!={name}' for name in ['mary','paul','fred','alice','john','susan','lucy']]
    assert z3_check(theory) == 'sat'


def test_person_typing_does_not_allow_negative_adjective_escape():
    assert z3_check([*background(), 'happy(mary)', '~(happy(mary)&person(mary))']) == 'unsat'


def test_multi_line_unit_is_one_stable_axiom():
    clauses = [{'id': 'p17', 'formula': '(p=>q)&(q=>r)', 'english': 'if p then q\nif q then r'}]
    rendered = problem([], clauses, 'r')
    assert rendered.count('fof(p17,') == 1
    assert '(p=>q)&(q=>r)' in rendered


@pytest.mark.parametrize('formula', ['p(X)', '![X]:(p(X)&q(Y))', 'p(mary)&p(mary,paul)'])
def test_parser_rejects_unbound_variables_and_mixed_arities(formula):
    with pytest.raises(ValueError):
        parse(formula)


@pytest.mark.parametrize('phrase,expected', [('is a musician','is not a musician'),
    ('can play chess','cannot play chess'), ('watches movies','does not watch movies'),
    ('has lived in France','has not lived in France'),
    ('regularly goes hiking','does not regularly go hiking')])
def test_lexical_negation_never_changes_logical_symbols(phrase, expected):
    assert negative(phrase) == expected


def test_no_one_does_not_negate_an_english_relative_clause(grammar):
    q = node(grammar, 'quantifier', formula=r'~\?')
    group = node(grammar, 'group', formula='room')
    quantifier = node(grammar, 'X_quantifier(quantifier,group)', q, group)
    conditional = node(grammar, 'cproperty(property,property)', prop(grammar), prop(grammar))
    body = node(grammar, 'X_property(cproperty)', conditional)
    sentence = node(grammar, 'term(X_quantifier,X_property)', quantifier, body)
    assert not sentence.check()


@pytest.mark.parametrize('quantifier,intended', [('!', '![X]:(room(X)=>preda(X))'),
                                                ('~!', '~![X]:(room(X)=>preda(X))')])
def test_universal_scope_is_unchanged(grammar, quantifier, intended):
    q = node(grammar, 'quantifier', formula=quantifier)
    group = node(grammar, 'group', formula='room')
    prefix = node(grammar, 'X_quantifier(quantifier,group)', q, group)
    body = node(grammar, 'X_property(property)', prop(grammar))
    sentence = node(grammar, 'term(X_quantifier,X_property)', prefix, body)
    equivalent(sentence @ 'tptp', intended)


def test_generation_is_independent_of_previous_groups():
    from gramforge.fol_nli.generate import make_group, digest
    first = make_group(44, 2, 4)
    make_group(44, 3, 4)
    assert digest(first) == digest(make_group(44, 2, 4))
    # Logical symbols stay stable, while lexical names are sampled per group.
    assert first['name_lexicon']
    assert all('pred' + chr(97 + i) in first['lexicon'] for i in range(26))


def test_identical_connective_operands_are_excluded(grammar):
    from gramforge.fol_nli.grammar import distinct_arguments
    left, right = prop(grammar), prop(grammar)
    assert not distinct_arguments([left, right])
    atom = node(grammar, 'atomic_predicate', formula='predb')
    different = node(grammar, 'property(atomic_predicate)', atom, formula='{0}(?)')
    assert distinct_arguments([left, different])
    for rule in grammar._instances:
        if rule.signature == 'property(property,property)':
            sentence = FastProduction(rule=rule, children=[left, right])
            assert not sentence.check()


def test_repeated_entities_excluded_without_making_sentiment_irreflexive(grammar):
    from gramforge.fol_nli.grammar import distinct_arguments, has_self_relation
    mary = node(grammar, 'entity(entity_)', node(grammar, 'entity_', formula='mary'))
    paul = node(grammar, 'entity(entity_)', node(grammar, 'entity_', formula='paul'))
    assert not distinct_arguments([mary, mary])
    assert distinct_arguments([mary, paul])
    comparative = node(grammar, 'comparative', formula='quieter')
    sentence = node(grammar, 'term(entity,comparative,entity)', mary, comparative, mary)
    assert not sentence.check()
    assert has_self_relation('~hate(mary,mary)')
    assert not has_self_relation('hate(mary,paul)')
    assert not has_self_relation('like(X,X)')
    assert z3_check([*background(), 'like(mary,mary)']) == 'sat'
