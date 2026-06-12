"""Utilities for building code-like grammars (not a grammar itself).

These factor the patterns that recur across `tinypy.py`, `mesopy.py`, and any
future code grammar (JavaScript, Rust, OOP languages, …):

  - lexical scoping (push/pop scope stack on function/class/method entry)
  - typed name binding with safe/all distinction (definitely vs maybe assigned)
  - block nesting (`nest_depth` for safe-var tracking, `loop_depth` for
    break/continue legality)
  - global definition registry (functions, classes, instances)
  - failure-rate gating
  - rendering a body node with state side-effects (`render_block`)

Grammar render lambdas mutate a single `CodeState` instance. Use the supplied
context managers inside `def`-style render functions; for lambdas, call
`render_block(node, state, loop=…)`.
"""

import random
from contextlib import contextmanager
from collections import defaultdict


class Scope:
    """One lexical scope: typed name bindings + `params` + `last` + free metadata.

    `all[t]` — every name currently bound to type `t` in this scope.
    `safe[t]` — subset of `all[t]` that is *definitely* assigned (params + assigns
                done at `nest_depth == 0` of this scope).
    `last` — most recently assigned name(s), used by `print(last_var)` etc.
    `meta` — open dict for grammar-specific extras (e.g. class methods/attrs).
    """
    __slots__ = ('kind', 'name', 'parent', 'params', 'all', 'safe', 'last', 'meta')

    def __init__(self, kind='function', name=None, parent=None,
                 params=(), ptypes=(), meta=None):
        self.kind, self.name, self.parent = kind, name, parent
        self.params = set(params)
        self.all  = defaultdict(set)
        self.safe = defaultdict(set)
        self.last = set()
        self.meta = dict(meta or {})
        for p, t in zip(params, ptypes):
            self.all[t].add(p); self.safe[t].add(p)

    def declare(self, name, typ, *, safe=False):
        """Bind `name` to type `typ`. Removes prior typing if name was reassigned."""
        for s in self.all.values():  s.discard(name)
        for s in self.safe.values(): s.discard(name)
        self.all[typ].add(name)
        if safe: self.safe[typ].add(name)
        self.last = {name}


class CodeState:
    """Render-time state for a code grammar.

    Pass one instance to your grammar's render lambdas (typically held in a
    closure). It is mutated as the tree is rendered DFS-left-to-right.
    """
    __slots__ = ('scopes', 'defs', 'nest_depth', 'loop_depth', 'aux')

    def __init__(self):
        self.scopes = []        # stack of Scope
        self.defs   = {}        # global name → dict (free metadata per definition)
        self.nest_depth = 0     # depth in any block (if/for/while/try/...)
        self.loop_depth = 0     # depth in loop bodies (for break/continue)
        self.aux = {}           # grab-bag (loop bounds, init stmts, etc.)

    @property
    def scope(self): return self.scopes[-1] if self.scopes else None

    def reset(self):
        self.scopes.clear(); self.defs.clear()
        self.nest_depth = self.loop_depth = 0
        self.aux.clear()

    @contextmanager
    def push_scope(self, *, kind='function', name=None, params=(), ptypes=(), meta=None):
        s = Scope(kind=kind, name=name, parent=self.scope,
                  params=params, ptypes=ptypes, meta=meta)
        self.scopes.append(s)
        prev_n, prev_l = self.nest_depth, self.loop_depth
        self.nest_depth = self.loop_depth = 0
        try:    yield s
        finally:
            self.scopes.pop()
            self.nest_depth, self.loop_depth = prev_n, prev_l

    @contextmanager
    def in_block(self, *, loop=False):
        self.nest_depth += 1
        if loop: self.loop_depth += 1
        try:    yield
        finally:
            self.nest_depth -= 1
            if loop: self.loop_depth -= 1

    def define(self, name, kind, **meta):
        """Register a globally-visible definition (function, class, …)."""
        self.defs[name] = {'kind': kind, **meta}


def pick_var(state, t, *, create=False, excluded=(), chars=None,
             safe_mode=None, literals=None):
    """Walk scope chain for a `t`-typed name.

    - `safe_mode` (or auto: True when `nest_depth==0`): prefer `safe[t]`; relax
      to `all[t]` if no safe match found anywhere up the chain.
    - `create=True`: when no name is found, fabricate a fresh char and return
      it (caller must register via `scope.declare` if it's an assignment).
    - `create=False`: when no name is found, call `literals[t]()` for a literal
      (typical for expression contexts — avoids phantom-var UnboundLocalError).
    - `excluded`: never return any name in this set (used for loop-counter
      protection).
    """
    if safe_mode is None: safe_mode = state.nest_depth == 0
    excluded = set(excluded)
    for scope in reversed(state.scopes):
        pool = (scope.safe if safe_mode else scope.all)[t] - excluded
        if pool: return random.choice(list(pool))
    if safe_mode:
        for scope in reversed(state.scopes):
            pool = scope.all[t] - excluded
            if pool: return random.choice(list(pool))
    if create:
        chars = chars or list("abcdefghijklmnopqrstuvwxyz")
        outer = state.scopes[0]
        taken = set().union(*outer.all.values()) | excluded
        return next((c for c in chars if c not in taken),
                    next((c for c in chars if c not in excluded), random.choice(chars)))
    if literals is None: return None
    return literals[t]() if callable(literals[t]) else literals[t]


def gated(rate, safe=None, risky=None):
    """`gated(p)` → bool w.p. `p`. `gated(p, safe, risky)` picks one (call if callable)."""
    hit = random.random() < rate
    if safe is None and risky is None: return hit
    pick = risky if hit else safe
    return pick() if callable(pick) else pick


def render_block(node, state, *, lang='py', loop=False):
    """Render a body node inside one nesting level. Equivalent to `state.in_block()`
    around `node.render(lang)` — convenient for use inside grammar lambdas."""
    with state.in_block(loop=loop):
        return node.render(lang)
