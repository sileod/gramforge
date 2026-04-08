import re
import unittest

from gramforge.generate import generate
from gramforge.grammars import simple_english_grammar


class EnglishGrammarQualityTest(unittest.TestCase):
    def test_extensive_generation_smoke_and_surface_sanity(self):
        grammar = simple_english_grammar(cap=6, questions=True)
        for seed in range(250):
            sentence = generate(grammar.start(), depth=8, seed=seed) @ 'eng'
            self.assertTrue(sentence, f'empty output for seed {seed}')
            self.assertNotIn('  ', sentence, f'double whitespace for seed {seed}: {sentence!r}')
            self.assertRegex(sentence, r'[.?]$', f'missing terminal punctuation for seed {seed}: {sentence!r}')

    def test_article_and_pp_constraints(self):
        grammar = simple_english_grammar(cap=6, questions=True)
        vowel_starts = ('artist', 'engineer', 'open', 'honest', 'odd')
        consonant_starts = ('cat', 'dog', 'scientist', 'student', 'teacher', 'friend', 'happy', 'sad', 'kind', 'quiet', 'brave', 'curious', 'friendly')
        locative_pronoun_pp = re.compile(r'\b(in|on|under) (him|her|it|them|us)\b')

        for seed in range(400):
            sentence = generate(grammar.start(), depth=8, seed=10_000 + seed) @ 'eng'
            lowered = sentence.lower()

            for token in vowel_starts:
                self.assertNotIn(f'a {token}', lowered, f'incorrect article in seed {seed}: {sentence!r}')
            for token in consonant_starts:
                self.assertNotIn(f'an {token}', lowered, f'incorrect article in seed {seed}: {sentence!r}')

            self.assertIsNone(
                locative_pronoun_pp.search(lowered),
                f'awkward locative PP+pronoun combination in seed {seed}: {sentence!r}',
            )
            self.assertNotIn('teach ', lowered, f'unexpected removed ditransitive verb in seed {seed}: {sentence!r}')

    def test_higher_depth_generation_stays_well_formed(self):
        grammar = simple_english_grammar(cap=6, questions=True)
        for seed in range(150):
            sentence = generate(grammar.start(), depth=12, seed=20_000 + seed) @ 'eng'
            self.assertTrue(sentence, f'empty high-depth output for seed {seed}')
            self.assertNotIn('  ', sentence, f'double whitespace at high depth for seed {seed}: {sentence!r}')
            self.assertRegex(sentence, r'[.?]$', f'missing terminal punctuation at high depth for seed {seed}: {sentence!r}')


if __name__ == '__main__':
    unittest.main()
