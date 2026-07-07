from gramforge.generate import generate
from gramforge.grammars import simple_english_grammar

grammar = simple_english_grammar(cap=6, questions=True)

print("=== Sample sentences ===")
for seed in range(80):
    s = generate(grammar.start(), depth=8, min_depth=4, seed=seed) @ 'eng'
    print(f"[{seed:3d}] {s}")
