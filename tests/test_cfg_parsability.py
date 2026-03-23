import importlib.util
import random
import re
import sys
import types
import unittest
from itertools import islice


HAVE_NLTK = importlib.util.find_spec('nltk') is not None


def install_runtime_stubs():
    easydict = types.ModuleType('easydict')
    class EasyDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
    easydict.EasyDict = EasyDict
    sys.modules.setdefault('easydict', easydict)

    numpy = types.ModuleType('numpy')
    numpy.floor = lambda x: int(x // 1)
    sys.modules.setdefault('numpy', numpy)

    for name in ('pandas', 'psutil'):
        sys.modules.setdefault(name, types.ModuleType(name))

    tqdm = types.ModuleType('tqdm')
    tqdm_auto = types.ModuleType('tqdm.auto')
    passthrough = lambda x=None, *args, **kwargs: x if x is not None else []
    tqdm.tqdm = passthrough
    tqdm_auto.tqdm = passthrough
    sys.modules.setdefault('tqdm', tqdm)
    sys.modules.setdefault('tqdm.auto', tqdm_auto)

    import anytree
    if not hasattr(anytree, 'LightNodeMixin'):
        anytree.LightNodeMixin = anytree.NodeMixin


install_runtime_stubs()

if HAVE_NLTK:
    from nltk.parse.earleychart import EarleyChartParser

    from gramforge import gramforge_to_nltk
    from gramforge.generate_sequential import generate_sequential
    from gramforge.generate_sequential_opt import generate_sequential_opt
    from gramforge.grammars import arith_grammar, dyck_grammar, simple_english_grammar


TOKEN_RE = re.compile(r"<=|>=|!=|==|\w+|[^\w\s]")
if HAVE_NLTK:
    CFG_CASES = [
        ('arith', lambda: arith_grammar(), 'py', {'min_depth': 4, 'max_depth': 6}),
        ('dyck', lambda: dyck_grammar(include_unicode=False), 'dyck', {'min_depth': 4, 'max_depth': 6}),
        ('english', lambda: simple_english_grammar(questions=False), 'eng', {'min_depth': 4, 'max_depth': 6}),
    ]
    GENERATORS = {
        'sequential': generate_sequential,
        'sequential_opt': generate_sequential_opt,
    }
else:
    CFG_CASES = []
    GENERATORS = {}


def tokenize(text):
    return TOKEN_RE.findall(text)


@unittest.skipUnless(HAVE_NLTK, 'nltk is required for CFG parsability checks')
class CFGParsabilityTest(unittest.TestCase):
    def test_generated_cfg_strings_are_parsable(self):
        for grammar_name, factory, lang, depth_kwargs in CFG_CASES:
            grammar = factory()
            parser = EarleyChartParser(gramforge_to_nltk(grammar))
            for generator_name, generator in GENERATORS.items():
                for seed in range(10):
                    random.seed(seed)
                    result = generator(grammar.start(), max_steps=5000, bushiness=0.8, **depth_kwargs)
                    self.assertTrue(result, f'{grammar_name}/{generator_name} failed to generate for seed {seed}')
                    rendered = result[0] @ lang
                    parses = list(islice(parser.parse(tokenize(rendered)), 1))
                    self.assertTrue(
                        parses,
                        f'{grammar_name}/{generator_name} produced unparsable output: {rendered!r}',
                    )


if __name__ == '__main__':
    unittest.main()
