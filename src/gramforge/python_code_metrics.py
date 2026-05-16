"""Metrics for generated Python code.

Pairs with grammars like `pygram` that produce executable Python: lets you
filter/score samples by static shape AND runtime behavior.

Usage:
    from gramforge.python_code_metrics import analyze, summarize

    report = analyze(code)                  # one sample
    table  = summarize([analyze(c) for c in samples])  # batch stats

The runtime portion uses sys.settrace with a SIGALRM timeout. A minimal
builtins set is used by default; pass your own dict to allow more.
"""

import ast
import signal
import statistics
import sys
from dataclasses import dataclass, asdict, field
from typing import Optional

__all__ = ['ExecutionReport', 'analyze', 'summarize',
           'SAFE_BUILTINS', 'format_table']


# A minimal builtins set sufficient for typical generated Python (functions,
# classes with inheritance + dunders, try/except). Extend if your grammar
# emits other globals.
SAFE_BUILTINS = {
    'print': lambda *a, **kw: None, 'range': range, 'len': len,
    'int': int, 'str': str, 'list': list, 'bool': bool, 'dict': dict,
    'Exception': Exception, 'ValueError': ValueError,
    'super': super, 'isinstance': isinstance,
    '__build_class__': __build_class__, '__name__': '__main__',
}


@dataclass
class ExecutionReport:
    """Static + runtime metrics for one generated Python sample."""
    # --- static ----------------------------------------------------------
    loc: int = 0                          # non-blank source lines
    parsed: bool = False
    syntax_error: Optional[str] = None
    defs: int = 0                         # `def` count, excluding __init__
    trivial_defs: int = 0                 # defs whose body is just `return name|const|None`
    dead_assigns: int = 0                 # locals reassigned without intervening read
    # --- runtime ---------------------------------------------------------
    runnable: bool = False
    runtime_error: Optional[str] = None
    exec_lines: int = 0                   # distinct lines touched at runtime
    steps: int = 0                        # total line-events (loop iterations count)
    timed_out: bool = False

    @property
    def density(self) -> float:           # what fraction of source was hit
        return self.exec_lines / self.loc if self.loc else 0.0

    @property
    def work(self) -> float:              # how many statement-executions per source line
        return self.steps / self.loc if self.loc else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['density'] = self.density
        d['work']    = self.work
        return d


def analyze(code: str, builtins: Optional[dict] = None,
            timeout_seconds: float = 2.0) -> ExecutionReport:
    """Run static + runtime analysis on `code`. Returns an ExecutionReport.

    `timeout_seconds` bounds the runtime trace (SIGALRM). Set to 0 to skip
    execution entirely (static-only report)."""
    r = ExecutionReport(loc=_count_loc(code))
    try:
        tree = ast.parse(code)
        r.parsed = True
    except SyntaxError as e:
        r.syntax_error = str(e); return r

    r.defs, r.trivial_defs = _count_trivial_defs(tree)
    r.dead_assigns         = _count_dead_assigns(tree)

    if timeout_seconds > 0:
        r.exec_lines, r.steps, err = _trace_exec(code, builtins or SAFE_BUILTINS,
                                                 timeout_seconds)
        r.runnable     = (err is None)
        r.runtime_error = None if err is None else type(err).__name__
        r.timed_out    = isinstance(err, TimeoutError)
    return r


def summarize(reports: list) -> dict:
    """Aggregate a batch of ExecutionReports into avg/median/min/max stats."""
    if not reports: return {}
    n = len(reports)
    numeric = ['loc', 'defs', 'trivial_defs', 'dead_assigns',
               'exec_lines', 'steps', 'density', 'work']
    out = {'n': n,
           'parsed':   sum(1 for r in reports if r.parsed),
           'runnable': sum(1 for r in reports if r.runnable),
           'timed_out': sum(1 for r in reports if r.timed_out)}
    errors = [r.runtime_error for r in reports if r.runtime_error]
    out['error_breakdown'] = {e: errors.count(e) for e in sorted(set(errors))}
    for k in numeric:
        vals = [getattr(r, k) if not k.endswith(('density', 'work'))
                else getattr(r, k) for r in reports]
        out[f'{k}_avg']    = statistics.mean(vals)
        out[f'{k}_median'] = statistics.median(vals)
        out[f'{k}_min']    = min(vals)
        out[f'{k}_max']    = max(vals)
    return out


def format_table(reports: list, header: bool = True) -> str:
    """Pretty-print per-sample metrics as a fixed-width table (one row per report)."""
    lines = []
    if header:
        lines.append(f"{'#':>3} {'loc':>4} {'def':>3} {'triv':>4} {'dead':>4} "
                     f"{'exec':>4} {'steps':>5} {'dens':>5} {'work':>5} {'run':>4} {'err':>14}")
    for i, r in enumerate(reports):
        if not r.parsed:
            lines.append(f"{i:>3} SYNTAX:{(r.syntax_error or '')[:60]}"); continue
        err = (r.runtime_error or '-')[:14]
        lines.append(f"{i:>3} {r.loc:>4} {r.defs:>3} {r.trivial_defs:>4} "
                     f"{r.dead_assigns:>4} {r.exec_lines:>4} {r.steps:>5} "
                     f"{r.density:>5.2f} {r.work:>5.2f} {str(r.runnable):>4} {err:>14}")
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------

def _count_loc(code: str) -> int:
    return sum(1 for l in code.split('\n') if l.strip())


def _count_trivial_defs(tree: ast.AST) -> tuple:
    """Return (n_defs, n_trivial). 'Trivial' = body is just `return X` for X
    in {name, constant, None}. `__init__` is skipped (it's never the focus).
    Disguised-identity bodies like `return x * 1` are NOT counted as trivial —
    they execute real ops and are useful as decode-this training material."""
    n_def = n_triv = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef): continue
        if node.name == '__init__': continue
        n_def += 1
        body = [s for s in node.body if not isinstance(s, ast.Pass)]
        if len(body) == 1 and isinstance(body[0], ast.Return):
            v = body[0].value
            if v is None or isinstance(v, (ast.Constant, ast.Name)):
                n_triv += 1
    return n_def, n_triv


def _count_dead_assigns(tree: ast.AST) -> int:
    """Count top-level `x = …` statements within function bodies whose target
    is reassigned before any intervening read of that name. Sparse dead
    assignments (≤ 1/method) provide useful LLM context filter; this is just
    a magnitude indicator so the caller can flag heavy slop."""
    dead = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef): continue
        pending = {}   # name -> True if assigned but unread since
        for stmt in node.body:
            reads = {n.id for n in ast.walk(stmt)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            if isinstance(stmt, ast.Assign):
                tgts = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                for t in tgts:
                    if t in pending and t not in reads: dead += 1
                    pending[t] = True
            for r in reads: pending.pop(r, None)
    return dead


def _trace_exec(code: str, builtins: dict, timeout: float) -> tuple:
    """Exec `code` with line tracing. Returns (distinct_lines, total_steps, err).
    err is None on success, the exception otherwise."""
    hits = set(); steps = [0]
    def tracer(frame, event, arg):
        if event == 'line':
            hits.add(frame.f_lineno); steps[0] += 1
        return tracer
    def _alarm(sig, frame): raise TimeoutError("execution timed out")
    err = None
    prev = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    sys.settrace(tracer)
    try:
        exec(compile(code, '<sample>', 'exec'), {'__builtins__': builtins})
    except BaseException as e:
        err = e
    finally:
        sys.settrace(None)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)
    return len(hits), steps[0], err
