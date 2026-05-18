"""Metrics for generated Python code.

Pairs with grammars like `pygram` that produce executable Python: lets you
filter/score samples by static shape AND runtime behavior.

Usage:
    from gramforge.metrics.python_code import analyze, summarize

    report = analyze(code)                            # one sample
    table  = summarize([analyze(c) for c in samples]) # batch stats
    report = analyze(code, sandbox='subprocess')      # real isolation

Field names align with reasoning-core's ExecutionResult (success, stdout,
error_type, error_msg, captured, timed_out, elapsed_ms) so the two are
interchangeable in downstream filters.
"""

import ast
import contextlib
import io
import multiprocessing
import signal
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

__all__ = ['ExecutionReport', 'analyze', 'summarize',
           'SAFE_BUILTINS', 'format_table']


SAFE_BUILTINS = {
    'print': print, 'range': range, 'len': len,
    'int': int, 'str': str, 'list': list, 'bool': bool, 'dict': dict,
    'Exception': Exception, 'ValueError': ValueError,
    'super': super, 'isinstance': isinstance,
    '__build_class__': __build_class__, '__name__': '__main__',
}


@dataclass
class ExecutionReport:
    """Static + runtime metrics for one generated Python sample.

    Runtime fields are a superset of reasoning-core's ExecutionResult."""
    # --- static ----------------------------------------------------------
    loc: int = 0
    parsed: bool = False
    syntax_error: Optional[str] = None
    defs: int = 0
    trivial_defs: int = 0                 # AST-trivial: body is `return name|const|None`
    dead_assigns: int = 0
    # --- runtime (names match reasoning-core's ExecutionResult) ----------
    success: bool = False
    stdout: str = ''
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    timed_out: bool = False
    elapsed_ms: float = 0.0
    captured: dict = field(default_factory=dict)
    # --- runtime (ours) --------------------------------------------------
    exec_lines: int = 0
    steps: int = 0
    n_calls: int = 0                      # function-call 'hops' during execution
    entry_args: Optional[list] = None     # args used in `_result = entry(…)` (if extracted)

    # back-compat aliases for our earlier API
    @property
    def runnable(self) -> bool: return self.success
    @property
    def runtime_error(self) -> Optional[str]: return self.error_type

    @property
    def density(self) -> float:
        return self.exec_lines / self.loc if self.loc else 0.0
    @property
    def work(self) -> float:
        return self.steps / self.loc if self.loc else 0.0

    @property
    def returned_input(self) -> bool:
        """True iff the entry function returned one of its input arguments —
        catches disguised-identity bodies like `return x * 1` that the static
        trivial_defs check misses."""
        if not self.success or self.entry_args is None: return False
        r = self.captured.get('_result')
        return r is not None and r in self.entry_args

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(density=self.density, work=self.work,
                 returned_input=self.returned_input)
        return d


def analyze(code: str, *, entry: str = 'f0', builtins: Optional[dict] = None,
            timeout_seconds: float = 2.0,
            sandbox: str = 'inprocess') -> ExecutionReport:
    """Static + runtime analysis of `code`.

    entry  — function name whose `_result = entry(…)` line we look for; the
             args parsed from that line are stored in `entry_args` to enable
             the `returned_input` identity check. Set to None to skip.
    sandbox — 'inprocess' (fast, SIGALRM timeout) or 'subprocess'
             (~50ms overhead, real isolation, survives infinite C-loops).
    """
    r = ExecutionReport(loc=_count_loc(code))
    try:
        tree = ast.parse(code); r.parsed = True
    except SyntaxError as e:
        r.syntax_error = str(e); return r

    r.defs, r.trivial_defs = _count_trivial_defs(tree)
    r.dead_assigns         = _count_dead_assigns(tree)
    if entry:
        r.entry_args = _extract_entry_args(tree, entry)

    if timeout_seconds <= 0: return r
    capture = ['_result'] if entry else []
    runner = _run_subprocess if sandbox == 'subprocess' else _run_inprocess
    data = runner(code, builtins or SAFE_BUILTINS, timeout_seconds, capture)
    for k, v in data.items():
        if hasattr(r, k): setattr(r, k, v)
    return r


def summarize(reports: list) -> dict:
    """Aggregate a batch of ExecutionReports into avg/median/min/max stats."""
    if not reports: return {}
    n = len(reports)
    fields_ = ['loc', 'defs', 'trivial_defs', 'dead_assigns',
               'exec_lines', 'steps', 'elapsed_ms', 'density', 'work']
    out = {'n': n,
           'parsed':         sum(1 for r in reports if r.parsed),
           'runnable':       sum(1 for r in reports if r.success),
           'returned_input': sum(1 for r in reports if r.returned_input),
           'timed_out':      sum(1 for r in reports if r.timed_out)}
    errors = [r.error_type for r in reports if r.error_type]
    out['error_breakdown'] = {e: errors.count(e) for e in sorted(set(errors))}
    for k in fields_:
        vals = [getattr(r, k) for r in reports]
        out[f'{k}_avg']    = statistics.mean(vals)
        out[f'{k}_median'] = statistics.median(vals)
        out[f'{k}_min']    = min(vals)
        out[f'{k}_max']    = max(vals)
    return out


def format_table(reports: list, header: bool = True) -> str:
    """Pretty-print per-sample metrics as a fixed-width table."""
    lines = []
    if header:
        lines.append(f"{'#':>3} {'loc':>4} {'def':>3} {'triv':>4} {'dead':>4} "
                     f"{'exec':>4} {'steps':>5} {'dens':>5} {'work':>5} "
                     f"{'ret=in':>6} {'run':>4} {'err':>14}")
    for i, r in enumerate(reports):
        if not r.parsed:
            lines.append(f"{i:>3} SYNTAX:{(r.syntax_error or '')[:60]}"); continue
        err = (r.error_type or '-')[:14]
        lines.append(f"{i:>3} {r.loc:>4} {r.defs:>3} {r.trivial_defs:>4} "
                     f"{r.dead_assigns:>4} {r.exec_lines:>4} {r.steps:>5} "
                     f"{r.density:>5.2f} {r.work:>5.2f} "
                     f"{str(r.returned_input):>6} {str(r.success):>4} {err:>14}")
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Static helpers
# --------------------------------------------------------------------------

def _count_loc(code: str) -> int:
    return sum(1 for l in code.split('\n') if l.strip())


def _count_trivial_defs(tree: ast.AST) -> tuple:
    """AST-trivial only: body is just `return name|const|None`. Disguised-
    identity like `return x * 1` is intentionally NOT flagged — it executes
    real ops and is useful as decode-this LLM training material. Use the
    runtime `returned_input` property for the semantic identity check."""
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
    """Magnitude indicator — sparse dead assigns are useful LLM context."""
    dead = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef): continue
        pending = {}
        for stmt in node.body:
            reads = {n.id for n in ast.walk(stmt)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        if t.id in pending and t.id not in reads: dead += 1
                        pending[t.id] = True
            for rname in reads: pending.pop(rname, None)
    return dead


def _extract_entry_args(tree: ast.AST, entry: str) -> Optional[list]:
    """Find `_result = entry(...)` or `print(entry(...))` at the module level
    and literal-eval the args. Returns None if the call isn't found or args
    aren't literals."""
    def _from_call(call):
        if not (isinstance(call.func, ast.Name) and call.func.id == entry):
            return None
        try: return [ast.literal_eval(a) for a in call.args]
        except (ValueError, SyntaxError): return None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            args = _from_call(stmt.value)
            if args is not None: return args
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            outer = stmt.value
            # `print(entry(...))` legacy form
            if (isinstance(outer.func, ast.Name) and outer.func.id == 'print'
                    and len(outer.args) == 1 and isinstance(outer.args[0], ast.Call)):
                args = _from_call(outer.args[0])
                if args is not None: return args
    return None


# --------------------------------------------------------------------------
# Runtime helpers
# --------------------------------------------------------------------------

_UNSERIALIZABLE = '<unserializable>'

def _safe_serialize(value: Any) -> Any:
    """Return a picklable representation, or a sentinel — used to make
    captured values cross the subprocess boundary safely."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        try: return type(value)(_safe_serialize(v) for v in value)
        except Exception: return _UNSERIALIZABLE
    if isinstance(value, dict):
        try: return {k: _safe_serialize(v) for k, v in value.items()
                     if isinstance(k, (str, int, float, bool))}
        except Exception: return _UNSERIALIZABLE
    return _UNSERIALIZABLE


def _trace_run(code: str, exec_globals: dict, capture_vars: list) -> dict:
    """Inner: settrace + redirect stdout + run. Returns a dict (no I/O)."""
    hits = set(); steps = [0]; calls = [0]
    def tracer(frame, event, arg):
        if event == 'line':
            hits.add(frame.f_lineno); steps[0] += 1
        elif event == 'call':
            calls[0] += 1
        return tracer
    buf = io.StringIO()
    err_type = err_msg = None
    t0 = time.perf_counter()
    sys.settrace(tracer)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, '<sandbox>', 'exec'), exec_globals)
    except BaseException as e:
        err_type = type(e).__name__; err_msg = str(e)[:200]
    finally:
        sys.settrace(None)
    elapsed = (time.perf_counter() - t0) * 1000
    captured = {v: _safe_serialize(exec_globals.get(v)) for v in capture_vars}
    return {'success': err_type is None, 'stdout': buf.getvalue(),
            'error_type': err_type, 'error_msg': err_msg,
            'elapsed_ms': elapsed,
            'exec_lines': len(hits), 'steps': steps[0],
            'n_calls': calls[0],
            'captured': captured, 'timed_out': False}


def _run_inprocess(code: str, builtins: dict, timeout: float, capture_vars: list) -> dict:
    """SIGALRM-bounded exec in the current process — fast (~0ms overhead)
    but can be defeated by C-level infinite loops."""
    def _alarm(*a): raise TimeoutError(f"execution exceeded {timeout}s")
    prev = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        data = _trace_run(code, {'__builtins__': builtins}, capture_vars)
    except TimeoutError as e:
        data = {'success': False, 'stdout': '', 'error_type': 'TimeoutError',
                'error_msg': str(e), 'elapsed_ms': timeout * 1000,
                'exec_lines': 0, 'steps': 0, 'captured': {}, 'timed_out': True}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)
    return data


def _subproc_worker(code, capture_vars, q):
    """Run in child process; uses real builtins (no restriction — the process
    boundary is the isolation, not builtin-filtering)."""
    try:
        q.put(_trace_run(code, {}, capture_vars))
    except BaseException as e:
        q.put({'success': False, 'stdout': '',
               'error_type': type(e).__name__, 'error_msg': str(e)[:200],
               'elapsed_ms': 0.0, 'exec_lines': 0, 'steps': 0,
               'captured': {}, 'timed_out': False})


def _run_subprocess(code: str, builtins: dict, timeout: float, capture_vars: list) -> dict:
    """Real isolation via multiprocessing.Process — ~50ms spawn overhead but
    survives C-level infinite loops, segfaults, and rogue file I/O.
    The `builtins` arg is accepted for API symmetry but ignored (child
    process has its own builtins)."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_subproc_worker,
                                args=(code, list(capture_vars), q), daemon=True)
    t0 = time.perf_counter()
    p.start(); p.join(timeout=timeout)
    if p.is_alive():
        p.kill(); p.join()
        return {'success': False, 'stdout': '', 'error_type': 'TimeoutError',
                'error_msg': f'Exceeded {timeout}s',
                'elapsed_ms': (time.perf_counter() - t0) * 1000,
                'exec_lines': 0, 'steps': 0, 'captured': {}, 'timed_out': True}
    if not q.empty():
        return q.get_nowait()
    return {'success': False, 'stdout': '', 'error_type': 'ProcessError',
            'error_msg': 'Child terminated unexpectedly',
            'elapsed_ms': (time.perf_counter() - t0) * 1000,
            'exec_lines': 0, 'steps': 0, 'captured': {}, 'timed_out': False}
