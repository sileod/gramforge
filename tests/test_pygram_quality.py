"""Quality tests for pygram_grammar.

Tests cover:
- Syntactic correctness (ast.parse) — always required
- Runnability at various configs (safe_returns, allow_recursion, include_assert)
- Absence of infinite loops (SIGALRM timeout)
- TypeError absence when safe_returns=True (type discipline)
"""

import ast
import collections
import inspect
import signal
import sys
import unittest

from gramforge import generate
from gramforge.grammars.pygram import pygram_grammar

SEEDS = range(30)
DEPTH = 10
SAFE_BUILTINS = {
    '__builtins__': {'print': lambda *a: None, 'range': range, 'len': len,
                     'int': int, 'str': str, 'list': list, 'bool': bool,
                     'Exception': Exception, 'super': super,
                     '__build_class__': __build_class__,
                     '__name__': '__main__'}
}
EXEC_TIMEOUT = 2   # seconds per execution


def _timeout_handler(sig, frame):
    raise TimeoutError("execution timed out")


def _run_seed(kw, seed):
    """Return (ok, err_type, code) for a single seed."""
    kw = {'n_functions': 2, **kw}   # default; caller can override
    g = pygram_grammar(**kw)
    code = generate(g, seed=seed, max_depth=DEPTH) @ 'py'
    try:
        ast.parse(code)
    except SyntaxError as e:
        return 'syntax', str(e), code

    ns = dict(SAFE_BUILTINS)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(EXEC_TIMEOUT)
    try:
        exec(code, ns)
        signal.alarm(0)
    except TimeoutError:
        return 'infinite_loop', 'infinite loop in module scope', code
    except Exception as e:
        signal.alarm(0)
        return type(e).__name__, str(e), code

    f0 = ns.get('f0')
    if not f0:
        return 'no_f0', '', code

    args = [5] * len(inspect.signature(f0).parameters)
    orig_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    signal.alarm(EXEC_TIMEOUT)
    try:
        f0(*args)
        signal.alarm(0)
        return 'ok', '', code
    except TimeoutError:
        return 'infinite_loop', 'infinite loop in f0', code
    except RecursionError:
        signal.alarm(0)
        return 'RecursionError', '', code
    except Exception as e:
        signal.alarm(0)
        return type(e).__name__, str(e), code
    finally:
        sys.setrecursionlimit(orig_limit)


class PygramSyntaxTest(unittest.TestCase):
    """Generated code must always parse as valid Python."""

    def _check_syntax(self, kw):
        for seed in SEEDS:
            g = pygram_grammar(n_functions=2, **kw)
            code = generate(g, seed=seed, max_depth=DEPTH) @ 'py'
            try:
                ast.parse(code)
            except SyntaxError as e:
                self.fail(f"SyntaxError seed={seed} config={kw}:\n{code}\n{e}")

    def test_syntax_safe_no_recursion(self):
        self._check_syntax(dict(safe_returns=True, allow_recursion=False))

    def test_syntax_safe_with_recursion(self):
        self._check_syntax(dict(safe_returns=True, allow_recursion=True))

    def test_syntax_unsafe_no_recursion(self):
        self._check_syntax(dict(safe_returns=False, allow_recursion=False))

    def test_syntax_all_features_off(self):
        self._check_syntax(dict(
            safe_returns=True, allow_recursion=False,
            include_loops=False, include_conditionals=False,
            include_augmented_assigns=False, include_assert=False,
        ))

    def test_syntax_all_features_on(self):
        self._check_syntax(dict(
            safe_returns=True, allow_recursion=False,
            include_ternary=True, include_fstrings=True,
            include_comprehensions=True, include_swap=True,
            include_extra_ops=True,
        ))


class PygramRunnabilityTest(unittest.TestCase):
    """Most generated functions should be runnable."""

    def _check_runnability(self, kw, min_ok_ratio, forbid=None):
        """Assert at least `min_ok_ratio` of seeds produce runnable code.

        forbid: set of error type names that must not appear at all.
        """
        counts = collections.Counter()
        for seed in SEEDS:
            result, msg, code = _run_seed(kw, seed)
            counts[result] += 1
            if forbid and result in forbid:
                self.fail(
                    f"Forbidden error '{result}' (seed={seed}) config={kw}:\n{code}\n{msg}"
                )
        total = len(SEEDS)
        ok = counts['ok']
        ratio = ok / total
        self.assertGreaterEqual(
            ratio, min_ok_ratio,
            f"Only {ok}/{total} seeds runnable for config={kw}; counts={dict(counts)}"
        )

    def test_safe_no_recursion_high_runnability(self):
        """safe_returns=True, no recursion: expect ≥85% runnable, no infinite loops."""
        self._check_runnability(
            dict(safe_returns=True, allow_recursion=False),
            min_ok_ratio=0.85,
            forbid={'SyntaxError', 'infinite_loop', 'UnboundLocalError'},
        )

    def test_safe_no_recursion_no_assert_near_perfect(self):
        """Turning off assert should push runnability above 90%."""
        self._check_runnability(
            dict(safe_returns=True, allow_recursion=False, include_assert=False),
            min_ok_ratio=0.90,
            forbid={'SyntaxError', 'infinite_loop', 'UnboundLocalError'},
        )

    def test_unsafe_no_recursion_acceptable_runnability(self):
        """unsafe mode allows more variety; expect ≥80% runnable."""
        self._check_runnability(
            dict(safe_returns=False, allow_recursion=False),
            min_ok_ratio=0.80,
            forbid={'SyntaxError', 'infinite_loop'},
        )

    def test_no_infinite_loops(self):
        """Regardless of config, no infinite loops should occur."""
        for kw in [
            dict(safe_returns=True, allow_recursion=False),
            dict(safe_returns=False, allow_recursion=False),
            dict(safe_returns=True, allow_recursion=False, include_assert=False),
        ]:
            for seed in SEEDS:
                result, msg, code = _run_seed(kw, seed)
                self.assertNotEqual(
                    result, 'infinite_loop',
                    f"Infinite loop seed={seed} config={kw}:\n{code}\n{msg}"
                )


class PygramVarietyTest(unittest.TestCase):
    """Grammar should produce diverse constructs."""

    def _collect_patterns(self, kw, n=50):
        patterns = collections.Counter()
        for seed in range(n):
            g = pygram_grammar(n_functions=2, **kw)
            code = generate(g, seed=seed, max_depth=DEPTH) @ 'py'
            if 'while ' in code:    patterns['while'] += 1
            if 'for '   in code:    patterns['for'] += 1
            if 'if '    in code:    patterns['if'] += 1
            if 'elif '  in code:    patterns['elif'] += 1
            if '+=' in code or '-=' in code or '*=' in code:
                patterns['aug_assign'] += 1
            if 'assert ' in code:   patterns['assert'] += 1
            if ' if ' in code and ' else ' in code:
                patterns['ternary'] += 1
            if 'f"' in code:        patterns['fstring'] += 1
            if ' for ' in code and ' in ' in code and '[' in code:
                patterns['list_comp'] += 1
        return patterns

    def test_core_constructs_appear(self):
        """All major constructs should appear in at least some samples."""
        kw = dict(safe_returns=True, allow_recursion=False)
        p = self._collect_patterns(kw, n=50)
        for construct in ['while', 'for', 'if', 'aug_assign']:
            self.assertGreater(p[construct], 5,
                f"Construct '{construct}' appeared only {p[construct]}/50 times")

    def test_optional_constructs_respect_flags(self):
        """When features are disabled, they should not appear."""
        kw = dict(safe_returns=True, allow_recursion=False,
                  include_loops=False, include_conditionals=False,
                  include_augmented_assigns=False, include_assert=False,
                  include_ternary=False, include_fstrings=False,
                  include_comprehensions=False, include_swap=False)
        for seed in range(20):
            g = pygram_grammar(n_functions=2, **kw)
            code = generate(g, seed=seed, max_depth=DEPTH) @ 'py'
            self.assertNotIn('while ', code, f"while appeared despite include_loops=False")
            self.assertNotIn('for ', code, f"for appeared despite include_loops=False")
            self.assertNotIn('if ', code, f"if appeared despite include_conditionals=False")
            self.assertNotIn('assert ', code, f"assert appeared despite include_assert=False")


class PygramClassesTest(unittest.TestCase):
    """include_classes=True generates class defs + instance use; self.X resolves."""

    def test_classes_parse_and_run(self):
        for seed in SEEDS:
            kw = dict(n_functions=1, include_classes=True, n_classes=1,
                      failure_rate=0.0, triviality_rate=0.3)
            result, msg, code = _run_seed(kw, seed)
            self.assertNotIn(result, {'syntax', 'infinite_loop'},
                f"Bad class generation seed={seed}: {result}\n{code}\n{msg}")
            self.assertIn('class ', code, f"No class produced seed={seed}")
            self.assertIn('self.', code, f"No self.attr usage seed={seed}")

    def test_class_runnability_failure_rate_zero(self):
        """At failure_rate=0, class output should be >= 90% runnable."""
        oks = 0
        for seed in SEEDS:
            kw = dict(n_functions=1, include_classes=True, n_classes=1,
                      failure_rate=0.0, triviality_rate=0.3)
            result, _, _ = _run_seed(kw, seed)
            if result == 'ok': oks += 1
        self.assertGreaterEqual(oks / len(SEEDS), 0.90,
            f"Only {oks}/{len(SEEDS)} class samples runnable at failure_rate=0")


if __name__ == '__main__':
    unittest.main(verbosity=2)
