import re
import unittest

from gramforge.generate import generate
from gramforge.grammars import FOL_grammar


class FOLGrammarQualityTest(unittest.TestCase):
    def test_generation_smoke_and_naturalness(self):
        grammar = FOL_grammar(N_PREMS=6)
        bad_patterns = [
            r'\bthey is\b',
            r'\ba [aeiou]',
            r'\bpersons\b',
            r'\band are\b',
            r'~',
        ]

        for seed in range(300):
            sample = generate(grammar.start(), depth=8, min_depth=6, seed=seed)
            eng = sample @ 'eng'
            tptp = sample @ 'tptp'

            self.assertTrue(eng, f'empty english output for seed {seed}')
            self.assertTrue(tptp, f'empty tptp output for seed {seed}')
            self.assertNotIn('  ', eng, f'double whitespace for seed {seed}: {eng!r}')

            lowered = eng.lower()
            for pattern in bad_patterns:
                self.assertIsNone(
                    re.search(pattern, lowered),
                    f"matched anti-pattern {pattern!r} for seed {seed}: {eng!r}",
                )


if __name__ == '__main__':
    unittest.main()
