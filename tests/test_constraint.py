from gramforge import Constraint, generate, init_grammar


def test_constraint_uses_first_grammar_language():
    R = init_grammar(["py"], preprocess_template=lambda s: s)
    R("start(lhs, rhs)", "{0}{1}", constraint=Constraint("0∉1"))
    R("lhs", "x")
    R("rhs", "x + 1")

    lhs = generate(R.get_rules("lhs")[0])
    rhs = generate(R.get_rules("rhs")[0])

    assert Constraint("0∉1")([lhs, rhs]) is False


def test_constraint_works_with_non_english_first_language():
    R = init_grammar(["code", "eng"])
    R("child", "foo", "bar")
    R("container", "prefix_foo_suffix", "unrelated english")

    child = generate(R.get_rules("child")[0])
    container = generate(R.get_rules("container")[0])

    assert Constraint("0∉1")([child, container]) is False


def test_constraint_accepts_language_index():
    R = init_grammar(["code", "eng"])
    R("child", "foo", "bar")
    R("container", "unrelated_code", "prefix bar suffix")

    child = generate(R.get_rules("child")[0])
    container = generate(R.get_rules("container")[0])

    assert Constraint("0∉1")([child, container]) is True
    assert Constraint("0∉1", index=1)([child, container]) is False
