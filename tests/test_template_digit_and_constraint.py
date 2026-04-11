"""
Tests for template/digit handling, Substitution, Constraint, and FOL patterns.

Covers fragile areas:
  - default_preprocess_template (bare-digit → {n} wrapping, edge cases)
  - Substitution N[?←X] chain mechanics
  - Constraint ∉ substring checks
  - FOL quantifier / entity-property substitution end-to-end
"""

import re
import pytest
from gramforge.grammar import init_grammar, Constraint, Substitution, default_preprocess_template
from gramforge.generate import generate
from gramforge.grammars import FOL_grammar
from gramforge.generate_sequential import FastProduction as FP


# ── helpers ───────────────────────────────────────────────────────────────

class _Stub:
    """Minimal stand-in for a FastProduction node: .render(lang) → text."""
    def __init__(self, text): self._text = text
    def render(self, lang=None): return self._text


def _make_grammar(*rules):
    """
    Build a fresh grammar with langs=['eng','tptp'] and the given rules.
    Each element of `rules` is a tuple passed directly to R(*r).
    Returns (R, [r0, r1, ...]).
    """
    R = init_grammar(['eng', 'tptp'])
    created = [R(*r) for r in rules]
    return R, created


def _fp(rule):
    """Wrap an existing rule object in a terminal FastProduction node."""
    node = FP(rule=rule)
    node.children = []
    return node


def _node(rule, *child_nodes):
    """Build a FP parent node with `rule`, wiring in the given child nodes."""
    node = FP(rule=rule)
    node.children = list(child_nodes)
    for c in child_nodes:
        c.parent = node
    return node


# ── 1. default_preprocess_template ────────────────────────────────────────

class TestDefaultPreprocessTemplate:

    def test_bare_single_digit_wraps(self):
        assert default_preprocess_template('0') == '{0}'

    def test_bare_digits_in_sequence_each_wrap(self):
        assert default_preprocess_template('0 1') == '{0} {1}'
        assert default_preprocess_template('0 1 2') == '{0} {1} {2}'

    def test_back_arrow_skips_preprocessing(self):
        # Only templates that contain ← are left unchanged.
        for s in ['1[?←0]', '0[?←X] and 1', r'\?[X,Y]:((0[?←X])&(1[?←Y])&(2(X,Y)))']:
            assert default_preprocess_template(s) == s, f"Expected no change for {s!r}"

    def test_escaped_question_mark_without_arrow_is_preprocessed(self):
        # A template with \? but WITHOUT ← has its digits wrapped normally.
        # The backslash-? is just text to the preprocessor.
        s = r'\?[X]:(0(X)&(?))'
        assert default_preprocess_template(s) == r'\?[X]:({0}(X)&(?))'

    def test_callable_passes_through(self):
        f = lambda *a: 'x'
        assert default_preprocess_template(f) is f

    def test_plain_text_without_digits_is_unchanged(self):
        assert default_preprocess_template('hello world') == 'hello world'
        assert default_preprocess_template('is a person') == 'is a person'

    def test_multi_digit_number_wraps_as_single_placeholder(self):
        # '10' must become '{10}', not '{1}{0}' – the regex matches whole digit runs.
        assert default_preprocess_template('10') == '{10}'
        assert default_preprocess_template('10 11') == '{10} {11}'

    def test_already_braced_digit_is_double_wrapped(self):
        # '{0}' has no ← so the regex runs and matches the '0' inside the braces,
        # yielding '{{0}}'. When .format() is applied, '{{0}}' renders to the
        # *literal string* '{0}' — a footgun when mixing explicit {n} syntax with
        # bare-digit templates.
        result = default_preprocess_template('{0} text')
        assert result == '{{0}} text', (
            "Preprocessor re-wraps the digit inside '{}', producing '{{0}}' which "
            "formats back to a literal '{0}' string. Use bare digits only (no curly "
            "braces) unless the template contains ← to skip preprocessing."
        )


# ── 2. Substitution ────────────────────────────────────────────────────────

class TestSubstitution:
    """Substitution(template)(arg0, arg1, ...) where args may be strings or nodes."""

    def test_replaces_question_mark_with_variable_name(self):
        sub = Substitution('0[?←X]')
        assert sub('tall(?)') == 'tall(X)'

    def test_chain_zero_then_format(self):
        # '0[?←1]': replace ? in arg0 with literal '1',
        # wrap converts '1' → '{1}', then .format fills {1} with rendered arg1.
        sub = Substitution('0[?←1]')
        assert sub('tall(?)', 'mary') == 'tall(mary)'

    def test_two_independent_substitutions_in_one_template(self):
        sub = Substitution('0[?←A]&1[?←B]')
        assert sub('p(?)', 'q(?)') == 'p(A)&q(B)'

    def test_multiple_question_marks_in_arg_all_replaced(self):
        sub = Substitution('0[?←X]')
        assert sub('f(?,?)') == 'f(X,X)'

    def test_escaped_question_mark_is_not_substituted_and_is_restored(self):
        # \? must survive the N[?←…] pass untouched and become a literal '?' at end.
        # Template: no N[?←...] match here, so inner_replaced is unchanged.
        # wrap converts bare '0' → '{0}', format fills with 'room'.
        # Final replace: \? → ?
        sub = Substitution(r'\?[X]:(0(X)&(?))')
        result = sub('room')
        assert result == r'?[X]:(room(X)&(?))'

    def test_fallback_to_format_when_no_substitution_pattern(self):
        # No N[?←...] pattern: wrap wraps digits, .format fills them.
        sub = Substitution('0 and 1')
        assert sub('alice', 'bob') == 'alice and bob'

    def test_digits_in_rendered_content_crash_substitution(self):
        """BUG: Substitution crashes when rendered content contains bare digits.

        wrap() converts ALL bare digits in the substituted text to {n} placeholders,
        including digits from content strings such as predicate names.
        A predicate 'pred5' gives 'pred5(X)' after substitution, but wrap() converts
        that to 'pred{5}(X)'. The subsequent .format() call needs positional arg 5
        but only 1 arg is present → IndexError.

        The current FOL grammar avoids this by naming predicates preda–predj
        (letters only), but the API offers no protection for arbitrary content.
        This test fails while the bug is present; it will pass once fixed.
        """
        sub = Substitution('0[?←X]')
        # Should produce 'pred5(X)' — the digit is part of the name, not a slot.
        result = sub('pred5(?)')
        assert result == 'pred5(X)'

    def test_substitution_with_node_arg(self):
        stub = _Stub('tall(?)')
        sub = Substitution('0[?←X]', lang='tptp')
        assert sub(stub) == 'tall(X)'


# ── 3. Constraint ──────────────────────────────────────────────────────────

class TestConstraint:

    def test_not_in_passes_when_disjoint(self):
        f = Constraint('0∉1')
        assert f([_Stub('mary'), _Stub('is tall')]) is True

    def test_not_in_fails_when_child0_present_in_child1(self):
        f = Constraint('0∉1')
        assert f([_Stub('mary'), _Stub('mary is tall')]) is False

    def test_bidirectional_fails_on_first_direction(self):
        f = Constraint('0∉1,1∉0')
        assert f([_Stub('old'), _Stub('bold old person')]) is False

    def test_bidirectional_passes_when_fully_disjoint(self):
        f = Constraint('0∉1,1∉0')
        assert f([_Stub('tall'), _Stub('kind')]) is True

    def test_three_way_constraint_catches_any_violation(self):
        f = Constraint('0∉1,1∉2,0∉2')
        clean = [_Stub('rich'), _Stub('old'), _Stub('happy')]
        dirty = [_Stub('rich'), _Stub('old'), _Stub('rich person')]  # 0∉2 violated
        assert f(clean) is True
        assert f(dirty) is False


# ── 4. Rule template rendering integration ───────────────────────────────

class TestRuleRenderIntegration:

    def test_terminal_rule_renders_its_template(self):
        _, [r] = _make_grammar(('greeting', 'hello', 'HELLO'))
        assert _fp(r).render('eng') == 'hello'
        assert _fp(r).render('tptp') == 'HELLO'

    def test_unary_rule_renders_child(self):
        _, [r_word, r_wrap] = _make_grammar(
            ('word', 'world', 'WORLD'),
            ('wrap(word)', '0', '0'),
        )
        node = _node(r_wrap, _fp(r_word))
        assert node.render('eng') == 'world'
        assert node.render('tptp') == 'WORLD'

    def test_binary_rule_renders_both_children_in_order(self):
        _, [ra, rb, r_pair] = _make_grammar(
            ('a', 'hello', 'H'),
            ('b', 'world', 'W'),
            ('c(a,b)', '0 1', '0-1'),
        )
        node = _node(r_pair, _fp(ra), _fp(rb))
        assert node.render('eng') == 'hello world'
        assert node.render('tptp') == 'H-W'

    def test_reversed_digit_order_swaps_children(self):
        _, [rx, ry, r_rev] = _make_grammar(
            ('x', 'first', 'A'),
            ('y', 'second', 'B'),
            ('z(x,y)', '1 0', '1-0'),   # deliberate reversal
        )
        node = _node(r_rev, _fp(rx), _fp(ry))
        assert node.render('eng') == 'second first'

    def test_callable_template_receives_child_nodes(self):
        _, [ra, rb, r_fn] = _make_grammar(
            ('a', 'x', 'x'),
            ('b', 'y', 'y'),
            ('c(a,b)', lambda a, b: f"<{a@'eng'},{b@'eng'}>", '0 1'),
        )
        node = _node(r_fn, _fp(ra), _fp(rb))
        assert node.render('eng') == '<x,y>'


# ── 5. FOL pattern unit tests ─────────────────────────────────────────────

class TestFOLPatternsUnit:
    """Test specific FOL template patterns using minimal hand-built grammars."""

    def test_entity_substituted_into_property_tptp(self):
        # langs=['tptp','eng'] so first template → tptp, second → eng
        R = init_grammar(['tptp', 'eng'])
        r_ent  = R('entity',           'mary',    'Mary')
        r_prop = R('property',         'tall(?)', 'is tall')
        # tptp: '1[?←0]' = render property (arg1), replace ? with literal '0',
        #        wrap turns '0' → '{0}', format fills {0} with entity.
        r_term = R('term(entity,property)', '1[?←0]', '0 1')
        node = _node(r_term, _fp(r_ent), _fp(r_prop))
        assert node.render('tptp') == 'tall(mary)'

    def test_entity_substituted_into_predicate(self):
        R = init_grammar(['tptp', 'eng'])
        r_ent  = R('entity',   'paul',    'Paul')
        r_prop = R('property', 'preda(?)', 'is preda')
        r_term = R('term(entity,property)', '1[?←0]', '0 1')
        node = _node(r_term, _fp(r_ent), _fp(r_prop))
        assert node.render('tptp') == 'preda(paul)'

    def test_x_quantifier_leaves_question_mark_for_parent(self):
        # X_quantifier renders with '?' intact so a parent Substitution can fill it.
        R = init_grammar(['tptp', 'eng'])
        r_q  = R('quantifier',              '!',    'everyone')
        r_g  = R('group',                   'room', 'in the room')
        r_xq = R('X_quantifier(quantifier,group)', '0[X]:(1(X)=>(?))','0 1')
        node = _node(r_xq, _fp(r_q), _fp(r_g))
        tptp = node.render('tptp')
        assert tptp == '![X]:(room(X)=>(?))', tptp
        assert '?' in tptp  # placeholder intact for parent substitution

    def test_universal_quantifier_full_chain(self):
        """term(X_quantifier, X_property) fills ? in X_quantifier with property."""
        R = init_grammar(['tptp', 'eng'])
        r_q   = R('quantifier',                    '!',          'everyone')
        r_g   = R('group',                          'room',       'in the room')
        r_xq  = R('X_quantifier(quantifier,group)', '0[X]:(1(X)=>(?))','0 1')
        r_adj  = R('adjective',                     'tall',       'tall')
        r_prop = R('property(adjective)',            '0(?)',       'is 0')
        r_xp   = R('X_property(property)',           '0[?←X]',    '0')
        r_term = R('term(X_quantifier,X_property)',  '0[?←1]',    '0 1')

        xp   = _node(r_xp,   _node(r_prop, _fp(r_adj)))
        xq   = _node(r_xq,   _fp(r_q),    _fp(r_g))
        term = _node(r_term, xq,           xp)

        assert term.render('tptp') == '![X]:(room(X)=>(tall(X)))'

    def test_existential_quantifier_chain(self):
        """E_quantifier fills ? in the existential template; E_property supplies the predicate."""
        R = init_grammar(['tptp', 'eng'])
        r_g   = R('group',                         'room',         'in the room')
        # Note: \?[X]:(...) has no ← → preprocessing runs and wraps '0' → '{0}'
        r_eq  = R('E_quantifier(group)',            r'\?[X]:(0(X)&(?))', 'someone 0')
        r_adj  = R('adjective',                    'tall',         'tall')
        r_prop = R('property(adjective)',           '0(?)',         'is 0')
        r_ep   = R('E_property(property)',          '0[?←X]',      '0')
        r_term = R('term(E_quantifier,E_property)', '0[?←1]',      '0 1')

        ep   = _node(r_ep,   _node(r_prop, _fp(r_adj)))
        eq   = _node(r_eq,   _fp(r_g))
        term = _node(r_term, eq, ep)

        assert term.render('tptp') == '?[X]:(room(X)&(tall(X)))'

    def test_adjective_chain_constraint_rejects_duplicate(self):
        """Constraint('1∉0') must block an adjective already in the chain."""
        R = init_grammar(['eng', 'tptp'])
        r_old       = R('adjective',                          'old(?)',   'old')
        r_chain_lf  = R('adjective_chain(adjective)',         '0(?)',     '0')
        r_chain_ext = R('adjective_chain(adjective_chain,adjective)',
                        '0&1(?)', '0 1', constraint=Constraint('1∉0'))

        chain = _node(r_chain_lf, _fp(r_old))
        ext   = _node(r_chain_ext, chain, _fp(r_old))  # same adjective repeated
        assert ext.check() is False

    def test_adjective_chain_constraint_allows_distinct(self):
        R = init_grammar(['eng', 'tptp'])
        r_old       = R('adjective',                          'old(?)',  'old')
        r_tall      = R('adjective',                          'tall(?)', 'tall')
        r_chain_lf  = R('adjective_chain(adjective)',         '0(?)',    '0')
        r_chain_ext = R('adjective_chain(adjective_chain,adjective)',
                        '0&1(?)', '0 1', constraint=Constraint('1∉0'))

        chain = _node(r_chain_lf, _fp(r_old))
        ext   = _node(r_chain_ext, chain, _fp(r_tall))  # distinct adjective
        assert ext.check() is True


# ── 6. FOL generation smoke tests ─────────────────────────────────────────

@pytest.fixture(scope='module')
def fol_grammar():
    return FOL_grammar(N_PREMS=6)


SEEDS = list(range(60))


@pytest.mark.parametrize('seed', SEEDS)
def test_fol_no_bare_question_mark_in_eng(fol_grammar, seed):
    sample = generate(fol_grammar.start(), depth=8, min_depth=5, seed=seed)
    eng = sample @ 'eng'
    assert '?' not in eng, f"Bare ? leaked into english (seed={seed}): {eng!r}"


@pytest.mark.parametrize('seed', SEEDS)
def test_fol_no_unresolved_placeholder_in_tptp(fol_grammar, seed):
    sample = generate(fol_grammar.start(), depth=8, min_depth=5, seed=seed)
    tptp = sample @ 'tptp'
    # '(?)' means a property placeholder was never filled by an entity.
    assert '(?)' not in tptp, f"Unresolved '(?)' in tptp (seed={seed}): {tptp!r}"


@pytest.mark.parametrize('seed', SEEDS)
def test_fol_tptp_has_no_double_whitespace(fol_grammar, seed):
    sample = generate(fol_grammar.start(), depth=8, min_depth=5, seed=seed)
    assert '  ' not in (sample @ 'tptp'), f"Double space in tptp (seed={seed})"


@pytest.mark.parametrize('seed', SEEDS)
def test_fol_constraint_no_free_x_without_quantifier(fol_grammar, seed):
    """Free variable X in tptp means the no_free_var constraint wasn't applied."""
    sample = generate(fol_grammar.start(), depth=8, min_depth=5, seed=seed)
    tptp = sample @ 'tptp'
    # If X appears, a universal [X]: or existential ?[...X...] must bind it.
    if 'X' in tptp:
        assert re.search(r'[\?\!]\[.*X.*\]', tptp), (
            f"Free variable X with no binding quantifier (seed={seed}): {tptp!r}"
        )
