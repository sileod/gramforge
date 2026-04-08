import ast
import unittest

from gramforge import generate
from gramforge.grammars import tinypy_grammar


class TinyPySyntaxTest(unittest.TestCase):
    def test_generated_programs_parse_as_python(self):
        levels = [None, "1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1"]
        for level in levels:
            grammar = tinypy_grammar(level=level)
            for seed in range(10):
                generated = generate(grammar, seed=seed, max_depth=12)
                code = generated @ "py"
                try:
                    ast.parse(code)
                except SyntaxError as exc:
                    self.fail(
                        f"tinypy(level={level!r}, seed={seed}) produced invalid Python:\n"
                        f"{code}\n{exc}"
                    )

    def test_generated_programs_execute_without_name_errors(self):
        levels = [None, "1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1"]
        safe_builtins = {"print": lambda *args, **kwargs: None, "range": range, "len": len}
        for level in levels:
            grammar = tinypy_grammar(level=level)
            for seed in range(10):
                generated = generate(grammar, seed=seed, max_depth=12)
                code = generated @ "py"
                try:
                    exec(code, {"__builtins__": safe_builtins}, {})
                except NameError as exc:
                    self.fail(
                        f"tinypy(level={level!r}, seed={seed}) produced undefined names:\n"
                        f"{code}\n{exc}"
                    )


if __name__ == "__main__":
    unittest.main()
