from gramforge.grammar import init_grammar


def test_default_realization_template_uses_space_separator():
    R = init_grammar(['eng'])
    rule = R('decl_simple(there, is, det_sg_a, n_sg_c)')

    assert rule.templates['eng'] == '{0} {1} {2} {3}'


def test_default_realization_template_supports_arbitrary_arity():
    R = init_grammar(['eng'])
    unary = R('wrap(item)')
    binary = R('pair(left, right)')

    assert unary.templates['eng'] == '{0}'
    assert binary.templates['eng'] == '{0} {1}'


def test_default_realization_template_separator_is_configurable():
    R = init_grammar(['eng'], default_separator=' | ')
    rule = R('triple(a, b, c)')

    assert rule.templates['eng'] == '{0} | {1} | {2}'

