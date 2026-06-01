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
            r'\ban \w+ people\b',
            r'\bpersons\b',
            r'\band are\b',
            r'\b([a-z]+) and \1 are respectively\b',
            r'\b[a-z]+_[a-z]+\b',
            r'\bis not [a-z-]+, not\b',
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

    def test_can_generate_without_setup(self):
        grammar = FOL_grammar(N_PREMS=4, include_setup=False)

        for seed in range(20):
            sample = generate(grammar.start(), depth=7, min_depth=4, seed=seed)
            eng = sample @ 'eng'
            tptp = sample @ 'tptp'

            self.assertNotIn('the only', eng)
            self.assertFalse(eng.startswith('there is a room.'))
            self.assertNotIn('there_is_a_room', tptp)

    def test_fixed_pronoun_modes_generate(self):
        for pronoun in ('she', 'he'):
            grammar = FOL_grammar(N_PREMS=4, pronoun=pronoun)
            for seed in range(20):
                sample = generate(grammar.start(), depth=7, min_depth=4, seed=seed)
                eng = sample @ 'eng'
                self.assertNotIn('he/she', eng)
                self.assertNotIn(f'{pronoun} are', eng)

        with self.assertRaises(ValueError):
            FOL_grammar(pronoun='it')


if __name__ == '__main__':
    unittest.main()
