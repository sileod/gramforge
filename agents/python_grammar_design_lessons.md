# Grammar design lessons — for transfer to other gramforge grammars

Distilled from building/refactoring `mesopy`. Most of these generalize to
any code-like grammar (JS, Rust, SQL, …); a few are Python-specific.

## What the framework gives you vs. what you must add yourself

### Framework (`gramforge.generate_sequential`) does
- Depth-bounded tree construction with pruning (`min_heights` / `max_heights`)
- Weighted rule selection
- Seed-reproducible randomness driven by a single thread
- Construction-time `state_constraint` / `constraint` predicates
- Multi-language render templates (`{0}`, `{1}`, or `0[?←x]` substitutions)

### Framework does NOT do
- **Render-time mutable state.** Your render lambdas must own a mutable
  state object. `Rule.state` is only injected at construction time and
  propagates via `ChainMap`; it's not enough for typed scope tracking.
- **State-aware rule selection.** `state_constraint` runs at construction,
  before render — it can't see `nest_depth`, `loop_depth`, or what's
  currently in scope. If you need that, do it inside the render lambda
  and emit a no-op fallback when the rule can't fire.
- **Symbol-table semantics.** Build your own; the framework treats render
  as opaque.

Implication: 60–70% of a real code grammar is **render-time machinery**,
not declarative rules.

## The six primitives that recur in every code grammar

If you start a new code grammar, expect to reinvent these. We extracted
them to `gramforge/codegen_utils/`:

1. **`Scope`** — typed name bindings with `all[t] ⊇ safe[t]` distinction.
   `safe[t]` is the definitely-assigned subset (top-level of function body,
   not inside any conditional/loop). Removing names from all type pools on
   each `declare()` keeps types exclusive.
2. **`CodeState`** — a stack of `Scope`s + a global definitions registry
   (`defs`) + nest/loop depth counters. Provides context managers for
   `push_scope(...)` and `in_block(loop=...)`.
3. **`pick_var(state, t, *, create=…)`** — walk the scope chain. For
   assignment targets, `create=True` fabricates a fresh name. For
   expressions, `create=False` falls back to a literal (this is essential
   — see "phantom variable" pitfall below).
4. **`gated(rate, safe, risky)`** — single-arg form returns a bool with
   probability `rate`; three-arg form picks one of two values/callables.
5. **`render_block(node, state, loop=…)`** — render a body node inside
   one nest level. The most-used helper.
6. A failure-rate gate.

These six cover everything mesopy needed. Stick to this list; if you
find yourself adding state to `S.aux` for grammar-specific tracking,
that's correct — `aux` is the open dict for one-off needs (loop math,
init values for var-reuse, etc.).

## Pitfalls and their fixes

### Phantom variables (UnboundLocalError epidemic)
Symptom: generated code references vars that are never assigned.
Root cause: when an expression rule (`EXPR_ID`) needs a typed var and
none exist, a naive picker fabricates a fresh name and returns it.
The name appears in the output but no assignment was ever generated
for it.

Fix: **distinguish creation from lookup**. Assignment-target rules can
create; expression rules must fall back to a literal. We did this with
one parameter: `pick_var(create=True)` for `VAR(CTX)`, `create=False`
elsewhere.

### Definitely-assigned vs. maybe-assigned (`safe` two-tier)
Inside `if`/`for`/`while` bodies, `x = …` doesn't make `x` "safe" — the
branch might not execute. But our scope tracking sees it as assigned.
If we later use `x` in the function's return, we get UnboundLocalError
at runtime.

Fix: track `safe[t]` separately and only add to it at `nest_depth == 0`.
When `safe_returns=True`, restrict returns and refs to `safe[t]`.

This is the single most valuable invariant in the grammar.

### Conditional safety still leaks across branches
Even with the safe two-tier, vars declared in an `if` body leak into the
`elif`/`else` condition lookup because the leaked var is in `all[t]`,
and our picker relaxes to `all` when `safe` is empty.

Fix (not yet applied in mesopy — open issue): in `if_chain`, snapshot
`scope.all` on body entry, restore on exit. Vars assigned inside a
branch don't escape.

### Nested-loop state collision
If your while-loop renderer stores loop-state in a flat dict (e.g.,
`state['loops']['var'] = 'x'`), an inner while clobbers the outer's
state. The outer loop's update line then references the wrong variable
(`7 = 7 + 2` is the syntax-error symptom).

Fix: save and restore the dict around the body render. Three lines.

### `pass` as fallback is slop
When a rule (swap, aug-assign) can't fire due to insufficient context,
returning `pass\n` produces visible runs of `pass\npass\npass`.

Fix: emit a useful no-op that also enriches scope, e.g.
`v = N` on a fresh int var. The next rule that needs an int now has
one to work with.

### `continue` in `while` body causes infinite loops
We emit the loop's update statement *after* the body. A `continue`
skips it — guaranteed infinite loop. (`break` is fine; it exits.)

Fix: track loop kind (for/while) in a stack. Allow `continue` only when
innermost is `for` (Python advances the iterator there automatically).

## Render-time vs. construction-time

The framework builds the tree first, then renders. Many "semantic"
decisions can only be made at render time (which variable is in scope,
whether we're inside a loop, etc.). This creates two patterns:

- **Construction-time choices**: rule weights, depth bounds, which
  feature flag is on. Static.
- **Render-time choices**: which name to pick, whether to emit a
  fallback, whether to gate by `failure_rate`. Dynamic.

Don't try to encode dynamic choices statically — you'll fight the
framework. Instead, let the grammar produce candidate rules and have
your render lambdas emit fallbacks when they can't fire.

## Steerable knobs that earn their keep

These knobs each unlock a real distribution shift in output. For a new
grammar, consider adding the same six:

| Knob | Effect |
|---|---|
| `safe_returns` | refs/returns restricted to definitely-assigned vars |
| `failure_rate` (0..1) | tautological asserts vs. random conditions; unguarded self-recursion suppressed at 0 |
| `triviality_rate` (0..1) | bias return picker toward compound (`v + w`, `v + literal`) vs. bare var |
| `usage_bias` (0..1) | down-weight already-referenced names so all defs get exercised |
| `inherit_rate` (0..1) | (for OO grammars) probability a class inherits from a previous one |
| `include_*` booleans | per-construct on/off (loops, conditionals, asserts, try/except, dunders, …) |

**Less helpful** (saw this in a sibling project): splitting `failure_rate`
into 5 sub-rates. The combination of `include_*` + `failure_rate` already
covers the practical configurations; per-source rates add API surface
without unlocking distinct behavior.

## Tests that actually catch regressions

Per-seed pass/fail is fine but blind. We get more from **aggregate
diagnostic tests** that:
- Generate N=30+ samples per config
- Compute summary stats with `gramforge.metrics.python_code.summarize()`
- Assert thresholds on `runnable`, `returned_input`, `steps_median`,
  `work_median`, `loc_median`, generation throughput

The single most useful figure is **"ms per useful sample"** — generation
time divided by count of `success AND not returned_input AND steps >= K`.
This composes all the quality dimensions into one number.

## Metrics that are worth computing

In `gramforge/metrics/python_code.py`, the `ExecutionReport` dataclass
has fields that earn their keep:

- `loc` — static size, sanity-check upper/lower bounds
- `defs`, `trivial_defs` — AST-trivial (return-name-or-const)
- `dead_assigns` — sparse-OK signal; dense means slop
- `success`, `error_type` — runnability
- `exec_lines`, `steps` — runtime coverage and step count
- `n_calls` — number of function-call "hops"
- `density` (`exec_lines/loc`), `work` (`steps/loc`) — derived
- `captured['_result']` + `entry_args` → `returned_input` property —
  catches "disguised identity" like `return x * 1` that AST-only
  triviality misses

Static "difficulty score" formulas (sum of height + 2·loops + …) are OK
as a fallback when execution isn't feasible, but **dynamic step count is
strictly better** when you can afford the exec.

## Sandboxing

`signal.SIGALRM` + `setitimer` works for in-process and is free, but
won't survive C-level infinite loops or rogue I/O. Add a
`multiprocessing.Process` sandbox option for scale data generation.
~50ms overhead per sample, real isolation.

## Don't modify the framework lightly

When you find yourself wanting a new `Rule.state` semantic or an
`on_enter` hook, ask: does this generalize to the other grammars (English,
FOL, tinypy)? If only your grammar needs it, build it in render lambdas.
We refactored mesopy entirely without touching `generate_sequential.py` or
`grammar.py`.

## File organization

- `gramforge/codegen_utils/` — primitives reusable across code grammars
- `gramforge/metrics/python_code.py` — Python-specific analysis (rename
  this for a different target language: `gramforge/js_code_metrics.py`,
  etc.). Aim for field-name overlap with downstream consumers'
  `ExecutionResult` dataclass if one exists.
- `gramforge/grammars/<lang>.py` — the actual grammar
- `tests/test_<lang>_quality.py` — per-seed correctness assertions
- `tests/test_<lang>_diagnostics.py` — aggregate threshold tests + a
  dashboard test that prints current numbers (no asserts)

## A short observation about the declarative/imperative split

The R() rules give you **syntactic skeleton** declaratively — what
statements exist, with what weights. That's genuinely valuable; reinventing
depth bounding + weighted choice imperatively would cost 100+ lines.

But the **semantic layer** (which var, what type, whether to recurse)
is 100% imperative inside render lambdas. The "declarative grammar"
framing is half-fiction here. Accept it; don't try to make semantics
declarative. The primitives in `gramforge.codegen_utils` just *name* the
imperative patterns so each grammar isn't reinventing them.
