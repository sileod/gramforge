"""Diagnostic tests for pygram_grammar — non-regression + steering signal.

Unlike test_pygram_quality which asserts pass/fail per seed, these tests
aggregate metrics across N samples per config and assert thresholds. The
goals:
  - Catch behavioral regressions (runnability, triviality, complexity drift)
  - Make the effect of each knob measurable in test form
  - Provide a generation-throughput benchmark

If a number drifts unexpectedly, the test will name it. To re-baseline,
read the numbers reported in the failure message and adjust the bound.
"""

import time
import unittest

from gramforge import generate
from gramforge.grammars.pygram import pygram_grammar
from gramforge.metrics.python_code import analyze, summarize

N_SAMPLES = 30   # per config
DEPTH = 14


def _gen_and_analyze(kw, n=N_SAMPLES, depth=DEPTH):
    """Generate n samples + analyze each. Returns (reports, generation_time_s)."""
    reports, t0 = [], time.perf_counter()
    for seed in range(n):
        g = pygram_grammar(**kw)
        code = generate(g, seed=seed, max_depth=depth) @ 'py'
        reports.append((code, analyze(code)))
    return reports, time.perf_counter() - t0


def _stats(reports):
    rs = [r for _, r in reports]
    s = summarize(rs)
    s['valid_nontrivial'] = sum(
        1 for r in rs
        if r.success and r.steps >= 10 and not r.returned_input
    )
    return s


class PygramSpeedTest(unittest.TestCase):
    """Generation throughput, including a 'per useful sample' figure that
    captures the rate of runnable AND non-trivial output."""

    def test_generation_throughput(self):
        kw = dict(n_functions=2, n_classes=2, include_classes=True,
                  failure_rate=0.0, triviality_rate=0.3, inherit_rate=0.5)
        reports, gen_t = _gen_and_analyze(kw, n=N_SAMPLES, depth=DEPTH)
        per_sample_ms  = 1000 * gen_t / N_SAMPLES
        s = _stats(reports)
        useful = s['valid_nontrivial']
        per_useful_ms = (1000 * gen_t / useful) if useful else float('inf')

        # Throughput floors — adjust if intentionally slowing down. Generous
        # so CI variance doesn't flake.
        self.assertLess(per_sample_ms, 400,
            f"Generation+analysis too slow: {per_sample_ms:.0f} ms/sample "
            f"(generated {N_SAMPLES} in {gen_t:.2f}s)")
        self.assertLess(per_useful_ms, 800,
            f"Per-useful sample too slow: {per_useful_ms:.0f} ms "
            f"(only {useful}/{N_SAMPLES} were valid+non-trivial)")


class PygramSyntaxTest(unittest.TestCase):
    """Parsability ceiling. Generated code must always parse as valid Python."""

    def test_all_configs_parse(self):
        configs = [
            ('safe',          dict(failure_rate=0.0)),
            ('fuzzed',        dict(failure_rate=1.0)),
            ('with_classes',  dict(include_classes=True, n_classes=2)),
            ('inheritance',   dict(include_classes=True, n_classes=3, inherit_rate=1.0)),
            ('no_features',   dict(include_loops=False, include_conditionals=False,
                                   include_augmented_assigns=False)),
        ]
        for label, extra in configs:
            kw = dict(n_functions=2, **extra)
            reports, _ = _gen_and_analyze(kw)
            s = _stats(reports)
            self.assertEqual(s['parsed'], N_SAMPLES,
                f"{label}: only {s['parsed']}/{N_SAMPLES} parsed")


class PygramRunnabilityTest(unittest.TestCase):
    """Runnability floors at known-safe configurations."""

    def test_safe_function_only_runs(self):
        kw = dict(n_functions=2, failure_rate=0.0, triviality_rate=0.3)
        s = _stats(_gen_and_analyze(kw)[0])
        self.assertGreaterEqual(s['runnable'], int(0.85 * N_SAMPLES),
            f"safe function-only: {s['runnable']}/{N_SAMPLES} runnable "
            f"(errors: {s['error_breakdown']})")

    def test_safe_with_classes_runs(self):
        kw = dict(n_functions=2, n_classes=2, include_classes=True,
                  failure_rate=0.0, triviality_rate=0.3, inherit_rate=0.5)
        s = _stats(_gen_and_analyze(kw)[0])
        self.assertGreaterEqual(s['runnable'], int(0.80 * N_SAMPLES),
            f"safe with classes: {s['runnable']}/{N_SAMPLES} runnable "
            f"(errors: {s['error_breakdown']})")

    def test_no_infinite_loops_at_safe(self):
        for kw in [dict(failure_rate=0.0),
                   dict(failure_rate=0.0, include_classes=True, n_classes=2)]:
            kw = dict(n_functions=2, **kw)
            s = _stats(_gen_and_analyze(kw)[0])
            self.assertEqual(s['timed_out'], 0,
                f"{kw}: {s['timed_out']} timeouts (should be 0 at failure_rate=0)")


class PygramTrivialityTest(unittest.TestCase):
    """triviality_rate should produce a measurable gradient of bare-return /
    identity-return frequency."""

    def test_triviality_rate_controls_identity_returns(self):
        # Endpoint targets get triviality_override=0, so we can't test the
        # knob on the endpoint's wrapped function. Disable endpoint and use
        # the legacy `_result = f0(args)` trailer to make returned_input
        # observable for f0 directly.
        results = {}
        for rate in [0.0, 1.0]:
            kw = dict(n_functions=2, failure_rate=0.0, triviality_rate=rate,
                      emit_endpoint=False, emit_result=True, print_result=True)
            s = _stats(_gen_and_analyze(kw)[0])
            results[rate] = s['returned_input']
        # At triviality_rate=1 we EXPECT identity returns; at 0 we expect few.
        self.assertGreater(results[1.0], results[0.0],
            f"triviality_rate has no effect: {results}")
        self.assertLessEqual(results[0.0] / N_SAMPLES, 0.20,
            f"triviality_rate=0 should yield <=20% identity returns, "
            f"got {results[0.0]}/{N_SAMPLES}")
        # The gap is the real signal — should be at least 3x
        self.assertGreater(results[1.0], 3 * results[0.0] + 1,
            f"triviality_rate gradient too weak: {results}")

    def test_ast_trivial_defs_are_sparse(self):
        """AST-trivial (return-name-or-const only) bodies should be rare."""
        kw = dict(n_functions=2, n_classes=2, include_classes=True,
                  failure_rate=0.0, triviality_rate=0.3, inherit_rate=0.5)
        s = _stats(_gen_and_analyze(kw)[0])
        # avg < 1 trivial def per sample
        self.assertLess(s['trivial_defs_avg'], 1.0,
            f"AST-trivial defs/sample = {s['trivial_defs_avg']:.2f} (should be <1)")


class PygramComplexityTest(unittest.TestCase):
    """Generated code should do *real* computational work, not be a 50-LOC
    return-input wrapper. The metric is runtime step count."""

    def test_steps_are_meaningful(self):
        kw = dict(n_functions=2, n_classes=2, include_classes=True,
                  failure_rate=0.0, triviality_rate=0.3, inherit_rate=0.5)
        s = _stats(_gen_and_analyze(kw)[0])
        # Median work (steps/loc) should indicate that at least half the
        # source lines get hit on average.
        self.assertGreaterEqual(s['work_median'], 0.6,
            f"work_median = {s['work_median']:.2f} (should be >=0.6)")
        # Steps median should be > LOC median: implies at least one loop
        # ran or methods were called multiple times.
        self.assertGreaterEqual(s['steps_median'], 20,
            f"steps_median = {s['steps_median']:.1f} (should be >=20)")

    def test_loc_in_expected_band(self):
        kw = dict(n_functions=2, n_classes=2, include_classes=True,
                  failure_rate=0.0, triviality_rate=0.3, inherit_rate=0.5)
        s = _stats(_gen_and_analyze(kw)[0])
        self.assertGreater(s['loc_median'], 20,
            f"loc_median = {s['loc_median']} (too small — code is shallow)")
        self.assertLess(s['loc_median'], 80,
            f"loc_median = {s['loc_median']} (too large — output bloated)")


class PygramDashboardTest(unittest.TestCase):
    """Always-printing diagnostic — not asserting anything, just dumping the
    current numbers to stdout so a maintainer can see drift at a glance.
    Run with `pytest -s` to see output."""

    def test_print_dashboard(self):
        configs = [
            ('fn only,  fr=0,   triv=0.3', dict(n_functions=2,
                failure_rate=0.0, triviality_rate=0.3)),
            ('fn only,  fr=0,   triv=1.0', dict(n_functions=2,
                failure_rate=0.0, triviality_rate=1.0)),
            ('+classes, fr=0,   triv=0.3', dict(n_functions=2, n_classes=2,
                include_classes=True, failure_rate=0.0,
                triviality_rate=0.3, inherit_rate=0.5)),
            ('+classes, fr=1,   triv=0.5', dict(n_functions=2, n_classes=2,
                include_classes=True, failure_rate=1.0,
                triviality_rate=0.5, inherit_rate=0.5)),
        ]
        print()
        hdr = (f"{'config':28}  {'parse':>5} {'run':>3} {'triv%':>5} "
               f"{'id%':>4} {'loc':>4} {'steps':>5} {'work':>4} {'ms/n':>5} {'ms/u':>5}")
        print(hdr); print('-' * len(hdr))
        for label, kw in configs:
            reports, gen_t = _gen_and_analyze(kw)
            s = _stats(reports)
            useful = s['valid_nontrivial']
            ms_n = 1000 * gen_t / N_SAMPLES
            ms_u = (1000 * gen_t / useful) if useful else 0
            print(f"{label:28}  "
                  f"{s['parsed']:>5} {s['runnable']:>3} "
                  f"{100*s['trivial_defs_avg']/max(1,s['defs_avg']):>4.0f}% "
                  f"{100*s['returned_input']/N_SAMPLES:>3.0f}% "
                  f"{s['loc_median']:>4.0f} "
                  f"{s['steps_median']:>5.0f} {s['work_median']:>4.2f} "
                  f"{ms_n:>4.0f}  {ms_u:>4.0f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
