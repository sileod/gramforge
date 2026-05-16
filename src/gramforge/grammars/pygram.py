import random
from .. import Substitution, Constraint, generate, init_grammar
from ..codegen_utils import CodeState, Scope, pick_var, gated, render_block

# Types: 'int', 'str', 'list'. State (S) holds a scope stack, a defs registry
# (S.defs[name] = {'kind':'function', 'arity':n, 'ret_t':T, 'ptypes':[…]}),
# and nest/loop depth counters. Each scope has `all[t]` (any-assigned) ⊇
# `safe[t]` (definitely-assigned at nest_depth==0). safe_returns=True
# restricts refs/returns to `safe`. STMTS / EXPRESSION are recursive →
# depth scales body length and arithmetic depth.


def pygram_grammar(
    max_number=16,
    # --- mode & structure ----------------------
    mode='function',            # 'program' = full script; 'function' = just defs
    n_functions=2,
    main_signature=None,        # ((ptype,...), ret_type) pins f0 signature
    f0_is_root=None,            # True → f_i only calls f_j for j >= i
    returns=True,
    type_hints=True,
    param_types=('int',),
    return_types=None,          # defaults to param_types
    n_outer_inits=0,
    max_params=2,
    # --- call graph ----------------------------
    allow_recursion=True,
    allow_cross_calls=True,
    # --- semantic quality ----------------------
    safe_returns=True,          # return only definitely-assigned vars
    min_body_stmts=1,
    failure_rate=0.5,           # 0=safe (tautological asserts, no unguarded self-rec); 1=anything
    triviality_rate=0.5,        # 0=force compound returns; 1=allow bare-var/literal returns
    usage_bias=0.6,             # 0=uniform name picks; 1=strongly prefer never-used names
                                # (down-weights already-referenced defs/vars in selection)
    # --- feature flags -------------------------
    include_print=True,
    include_loops=True,
    include_conditionals=True,
    include_augmented_assigns=True,
    include_ternary=True,
    include_assert=True,
    include_comprehensions=True,
    include_fstrings=True,
    include_extra_ops=True,
    include_swap=True,
    include_break_continue=True,
    include_try_except=True,
    include_classes=False,            # generate class definitions + instance usage
    n_classes=1,                      # only used when include_classes=True
):
    R = init_grammar(['py'])
    chars = list("abcdefghijklmnopqrstuvwxyz")
    param_types  = tuple(param_types)
    return_types = tuple(return_types) if return_types is not None else param_types
    if mode not in ('program', 'function'):
        raise ValueError(f"mode must be 'program' or 'function', got {mode!r}")
    if f0_is_root is None:
        f0_is_root = (mode == 'function')
    if mode == 'function' and n_functions < 1:
        raise ValueError("mode='function' requires n_functions >= 1")

    # === 1. State + literals ===
    S = CodeState()
    TYPES = ('int', 'str', 'list')
    LITERALS = {
        'int':  lambda: str(random.randint(0, max_number)),
        'str':  lambda: random.choice(['"hi"', '"cat"', '"go"', '"sun"']),
        'list': lambda: "[0, 1, 2]",
    }

    def reset_state(ctx):
        S.reset()
        S.scopes.append(Scope(kind='module'))
        S.aux.update({'fn_plan': [], 'fn_plan_idx': 0, 'current_fn': None, 'loops': {},
                      'class_plan': [], 'class_idx': 0, 'method_plan': [], 'method_idx': 0,
                      'use_counts': {}})
        if n_functions > 0: _make_plan()
        if include_classes and n_classes > 0: _make_class_plan()
        return ""

    def _bump_use(name):
        S.aux['use_counts'][name] = S.aux['use_counts'].get(name, 0) + 1

    def _biased_pick(names):
        """Pick from `names` with weight inversely proportional to prior uses.
        usage_bias=0 → uniform; usage_bias=1 → strongly prefer never-used names."""
        if not names: return None
        if usage_bias <= 0: return random.choice(names)
        weights = [1.0 / (1.0 + S.aux['use_counts'].get(n, 0) * usage_bias * 4) for n in names]
        return random.choices(names, weights=weights, k=1)[0]

    # === 2. Var/name primitives ===
    def loop_excl():     # protect current while-loop counter from re-assignment in body
        v = S.aux['loops'].get('var') if S.nest_depth > 0 else None
        return {v} if v else set()

    def pv(t, *, create=False):
        return pick_var(S, t, create=create, excluded=loop_excl(),
                        chars=chars, literals=LITERALS,
                        safe_mode=safe_returns and S.nest_depth == 0)

    def init_vals(): return S.scope.meta.setdefault('init_vals', {})

    def render_init(ctx):
        used = set().union(*S.scope.all.values()) | set(init_vals())
        v = random.choice([c for c in chars if c not in used] or chars)
        d = str(random.randint(0, max_number))
        init_vals()[v] = d
        S.scope.declare(v, 'int', safe=True)
        return f"{v} = {d}\n"

    def render_assign(v_node, e_node, kind='int'):
        v, e = v_node.render('py'), e_node.render('py')
        init_vals().setdefault(v, '0')
        S.scope.declare(v, kind, safe=(S.nest_depth == 0))
        return f"{v} = {e}\n"

    def get_assigned_var(ctx): return pv('int')
    def get_atom(ctx):
        if any(S.scope.all.values()) and random.random() < 0.7: return pv('int')
        return str(random.randint(0, max_number))
    def get_last_var(ctx):
        top = S.scope
        if top.last and not (safe_returns and S.nest_depth == 0):
            return next(iter(top.last))
        excl = loop_excl()
        for scope in reversed(S.scopes):
            for t in TYPES:
                pool = scope.safe[t] - excl
                if pool: return random.choice(list(pool))
        return pv('int')

    def render_cond_expr(v1, op, v2):
        lhs, op_s, rhs = v1.render('py'), op.render('py'), v2.render('py')
        if lhs == rhs and random.random() < 0.8:
            pool = (S.scope.safe if S.nest_depth == 0 else S.scope.all)['int']
            alts = [c for c in pool if c != lhs]
            if alts: rhs = random.choice(alts)
        return f"{lhs} {op_s} {rhs}"

    # === 3. Loops (math + while-var injection) ===
    def render_loop_math(ctx, mode):
        L = S.aux['loops']
        if mode == 'init':
            v = str(random.randint(0, 20)); L['val'] = v; return v
        init = int(L.get('val', '0'))
        step, count = random.choice([(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)])
        L['step'] = str(step)
        return str(step * count + init - 1) if mode == 'final_less' else \
               str(init - step * count + 1)

    def render_while_var(ctx):
        excl = loop_excl()
        v = None
        for scope in reversed(S.scopes):
            pool = scope.safe['int'] - excl
            if pool:
                v = random.choice(list(pool))
                S.aux['loops']['init_stmt'] = ''
                break
        if v is None:
            # No safe int available: create one with an explicit init before the loop.
            used = set().union(*S.scope.all.values()) | set(init_vals()) | excl
            v = next((c for c in chars if c not in used), random.choice(chars))
            init_vals()[v] = '0'
            S.scope.declare(v, 'int', safe=True)
            S.aux['loops']['init_stmt'] = f"{v} = 0\n"
        S.aux['loops']['var'] = v
        S.aux['loops']['val'] = init_vals().get(v, '0')
        return v

    def render_while_update(ctx, op):
        v, s = S.aux['loops'].get('var', 'i'), S.aux['loops'].get('step', '1')
        return f"{v} = {v} {op} {s}"

    # === 4. Structural ===
    def _indent(text, prefix='    '):
        return '\n'.join(prefix + l.replace('\t', '    ')
                         for l in text.split('\n') if l.strip())

    def if_chain(*pairs):  # (head_node, body_node)... → if/elif/else chain
        return ''.join(f"{h.render('py')}{_indent(render_block(b, S))}\n" for h, b in pairs)

    def _assign(kind='int'):
        return render_assign if kind == 'int' else lambda v, e: render_assign(v, e, kind=kind)

    # === 5. Typed calls ===
    def _fn_idx(name): return int(name[1:]) if name and name.startswith('f') else -1

    def _callable_functions(ret_t):
        candidates, current = [], S.aux['current_fn']
        cidx = _fn_idx(current)
        unguarded = S.nest_depth == 0
        for name, spec in S.defs.items():
            if spec.get('kind') != 'function' or spec.get('ret_t') != ret_t: continue
            is_self = (name == current)
            if is_self and not allow_recursion: continue
            if is_self and unguarded and not gated(failure_rate): continue
            if (not is_self) and current is not None and not allow_cross_calls: continue
            if f0_is_root and current is not None and _fn_idx(name) < cidx: continue
            candidates.append((name, spec['ptypes']))
        return candidates

    def _render_call_of_type(ret_t):
        cands = _callable_functions(ret_t)
        if cands:
            name = _biased_pick([c[0] for c in cands])
            ptypes = dict(cands)[name]
            _bump_use(name)
            return f"{name}({', '.join(_arg_of_type(t) for t in ptypes)})"
        return LITERALS[ret_t]()

    def _arg_of_type(t):
        safe_mode = safe_returns and S.nest_depth == 0
        for scope in reversed(S.scopes):
            pool = scope.safe[t] if safe_mode else scope.all[t]
            if pool and random.random() < 0.6: return random.choice(list(pool))
        if safe_mode:
            for scope in reversed(S.scopes):
                if scope.all[t] and random.random() < 0.6:
                    return random.choice(list(scope.all[t]))
        return LITERALS[t]()

    # === 6. Function defs ===
    def _make_plan():
        plan = []
        for i in range(n_functions):
            if i == 0 and main_signature is not None:
                ptypes, ret_t = main_signature; ptypes = list(ptypes); n = len(ptypes)
            else:
                n = random.randint(1, max(1, max_params))
                ptypes = [random.choice(list(param_types)) for _ in range(n)]
                ret_t  = random.choice(list(return_types))
            name = f"f{i}"
            S.define(name, 'function', arity=n, ret_t=ret_t, ptypes=ptypes)
            plan.append((name, n, ret_t, ptypes))
        S.aux['fn_plan'] = plan
        S.aux['fn_plan_idx'] = 0

    _DEFAULT_RET = {'int': '0', 'str': '""', 'list': '[]'}
    def _pick_return_value(ret_t):
        top = S.scope
        safe_pool = top.safe[ret_t]
        # Inside a method, treat enclosing class attrs as additional safe values.
        cls = next((s for s in reversed(S.scopes) if s.kind == 'class'), None) if include_classes else None
        if cls and ret_t == 'int':
            safe_pool = safe_pool | {f"self.{a}" for a, t in cls.meta.get('attrs', {}).items() if t == ret_t}
        recur_ok = allow_recursion and gated(failure_rate)
        r = random.random()
        if (r < 0.25 and recur_ok) or (r < 0.25 and allow_cross_calls):
            return _render_call_of_type(ret_t)
        if ret_t == 'int' and len(safe_pool) >= 2:
            if r < 0.40:
                a, b = random.sample(list(safe_pool), 2)
                return f"{a} {random.choice(['+', '-', '*'])} {b}"
            if r < 0.50 and include_ternary:
                a, b = random.sample(list(safe_pool), 2)
                cv = random.choice(list(safe_pool))
                return f"{a} if {cv} {random.choice(['<', '>', '==', '!='])} {random.randint(0, max_number)} else {b}"
        pool = safe_pool if safe_returns else top.all[ret_t]
        if pool and ret_t == 'int' and random.random() > triviality_rate:
            if len(pool) >= 2:
                a, b = random.sample(list(pool), 2)
                return f"{a} {random.choice(['+', '-', '*'])} {b}"
            a = next(iter(pool))
            return f"{a} {random.choice(['+', '-', '*'])} {random.randint(1, max_number)}"
        if top.last and (last := [v for v in top.last if v in pool]):
            return last[0]
        if pool: return random.choice(list(pool))
        if safe_returns and (pp := top.all[ret_t] & top.params):
            return random.choice(list(pp))
        return _DEFAULT_RET[ret_t]

    def render_func_def(body_node):
        idx = S.aux['fn_plan_idx']; S.aux['fn_plan_idx'] = idx + 1
        fname, n_params, ret_t, ptypes = S.aux['fn_plan'][idx]

        used = set().union(*S.scope.all.values()) | set(init_vals())
        pool = [c for c in chars if c not in used]
        params = (random.sample(pool, n_params) if len(pool) >= n_params
                  else random.sample(chars, n_params))

        prev_fn = S.aux.get('current_fn')
        S.aux['current_fn'] = fname
        with S.push_scope(kind='function', name=fname, params=params, ptypes=ptypes):
            body_text = body_node.render('py')
            ret_v = _pick_return_value(ret_t)
        S.aux['current_fn'] = prev_fn

        if type_hints:
            sig = ', '.join(f"{p}: {t}" for p, t in zip(params, ptypes))
            header = f"def {fname}({sig}) -> {ret_t}:" if returns else f"def {fname}({sig}):"
        else:
            header = f"def {fname}({', '.join(params)}):"
        tail = f"\n    return {ret_v}" if returns else ("" if body_text.strip() else "\n    pass")
        return f"{header}\n{_indent(body_text)}{tail}\n"

    # === 6b. Class defs (V0) ===
    # Convention: __init__ is auto-emitted (assigns 1-2 attrs); the grammar tree
    # only spans the non-init methods. `self.X` references resolve via the
    # enclosing class scope's meta['attrs']; outside a class, SELF_ATTR falls
    # back to a literal so the rule is safe to include in TERM globally.
    _ATTR_POOL = ['a', 'b', 'x', 'y', 'value', 'count']

    def _make_class_plan():
        plan = []
        for i in range(n_classes):
            cname = f"C{i}"
            attrs = {a: 'int' for a in random.sample(_ATTR_POOL, random.randint(1, 2))}
            methods = []
            for j in range(random.randint(1, 2)):
                n = random.randint(0, 1)
                methods.append({'name': f"m{j}",
                                'params': [random.choice([c for c in chars if c not in attrs])] if n else [],
                                'ptypes': ['int'] * n, 'ret_t': 'int'})
            plan.append({'name': cname, 'attrs': attrs, 'methods': methods})
            S.define(cname, 'class', attrs=attrs, methods=methods)
        S.aux['class_plan'] = plan
        S.aux['class_idx'] = 0

    def _class_scope():
        return next((s for s in reversed(S.scopes) if s.kind == 'class'), None)

    def render_self_attr(ctx):
        cls = _class_scope()
        if cls and cls.meta.get('attrs'):
            return f"self.{random.choice(list(cls.meta['attrs']))}"
        return LITERALS['int']()

    def render_method_def(body_node):
        idx = S.aux['method_idx']; S.aux['method_idx'] = idx + 1
        plan = S.aux['method_plan']
        if idx >= len(plan):  # grammar produced more methods than planned — extend
            n = random.randint(0, 1)
            plan.append({'name': f"m{idx}", 'params': ['p'] if n else [],
                         'ptypes': ['int'] * n, 'ret_t': 'int'})
        spec = plan[idx]
        mname, params, ptypes, ret_t = spec['name'], spec['params'], spec['ptypes'], spec['ret_t']

        prev_fn = S.aux.get('current_fn')
        S.aux['current_fn'] = mname
        with S.push_scope(kind='function', name=mname, params=params, ptypes=ptypes):
            body_text = body_node.render('py')
            ret_v = _pick_return_value(ret_t)
        S.aux['current_fn'] = prev_fn

        sig = ', '.join(['self'] + [f"{p}: {t}" for p, t in zip(params, ptypes)])
        header = f"def {mname}({sig}) -> {ret_t}:" if type_hints else f"def {mname}({sig}):"
        return f"{header}\n{_indent(body_text)}\n    return {ret_v}\n"

    def render_class_def(body_node):
        cidx = S.aux['class_idx']; S.aux['class_idx'] = cidx + 1
        spec = S.aux['class_plan'][cidx]
        cname, attrs = spec['name'], spec['attrs']

        # Plan can be extended by render_method_def (recursive METHODS); after
        # rendering we know exactly how many landed — prune `spec['methods']`
        # to that set so `render_instance_use` can't pick a non-existent name.
        S.aux['method_plan'] = list(spec['methods'])
        S.aux['method_idx']  = 0
        with S.push_scope(kind='class', name=cname, meta={'attrs': attrs}):
            methods_text = body_node.render('py')
        spec['methods'] = S.aux['method_plan'][:S.aux['method_idx']]
        S.defs[cname]['methods'] = spec['methods']

        init_sig  = ', '.join(['self'] + [f"{p}: {t}" for p, t in attrs.items()])
        init_body = '\n'.join(f"    self.{p} = {p}" for p in attrs)
        init_def  = f"def __init__({init_sig}):\n{init_body}\n"
        return f"class {cname}:\n{_indent(init_def + methods_text)}\n"

    def render_instance_use(ctx):
        """Emit one instantiate+method-call block per defined class.

        Bias picks each class's least-used method so the output exercises
        the class surface area rather than calling m0 on everything."""
        classes = [(n, s) for n, s in S.defs.items() if s.get('kind') == 'class']
        lines = []
        for cname, spec in classes:
            if not spec.get('methods'): continue
            init_args = ', '.join(_arg_of_type(t) for t in spec['attrs'].values())
            mname = _biased_pick([m['name'] for m in spec['methods']])
            method = next(m for m in spec['methods'] if m['name'] == mname)
            margs = ', '.join(_arg_of_type(t) for t in method['ptypes'])
            used = set().union(*S.scope.all.values()) | set(init_vals())
            var = next((c for c in chars if c not in used), 'c')
            S.scope.all[f'instance:{cname}'].add(var)
            init_vals()[var] = f"{cname}(...)"
            _bump_use(cname); _bump_use(mname)
            lines.append(f"{var} = {cname}({init_args})")
            lines.append(f"print({var}.{mname}({margs}))")
        return '\n'.join(lines) + ('\n' if lines else '')

    # ============ 7. Grammar rules ============
    R('CTX', '')
    R('RESET(CTX)', reset_state)

    # Terminals
    R('DIGIT',       lambda: str(random.randint(0, max_number)))
    R('SMALL_INDEX', lambda: str(random.randint(0, 1)))
    R('STR_LIT',     lambda: random.choice(['"hi"', '"cat"', '"go"', '"sun"']))
    R('LIST_LIT(DIGIT, DIGIT, DIGIT)', '[0, 1, 2]')
    R('VAR(CTX)',    lambda x: pv('int', create=True))
    R('BOOL_LIT',    lambda: random.choice(['True', 'False']))
    for _op in ['+', '-', '*']: R('ARITH_OP', _op)
    for _op in ['<', '>', '<=', '>=', '!=', '==']: R('REL_OP', _op)
    for _op in ['and', 'or']: R('LOG_INFIX', _op)
    R('LOG_PREFIX', 'not ')

    R('NON_ZERO_DIGIT', lambda: str(random.randint(1, max(1, max_number))))
    if include_extra_ops:
        R('DIV_OP', '//'); R('DIV_OP', '%')

    # Integer expressions (recursive EXPRESSION for depth scaling)
    R('EXPR_ID(CTX)', get_assigned_var)
    R('ATOM(CTX)',    get_atom)
    R('ATOM(BOOL_LIT)', '0', weight=0.15)
    R('TERM(EXPR_ID)', '0'); R('TERM(DIGIT)', '0'); R('TERM(ATOM)', '0')
    R('TERM(CALL_INT)', '0', weight=0.6)
    if include_classes:
        # SELF_ATTR resolves to `self.<attr>` inside a class method; a literal
        # otherwise (so the rule is safe to include in TERM/EXPR_ID globally).
        R('SELF_ATTR(CTX)', render_self_attr)
        R('TERM(SELF_ATTR)',    '0', weight=0.8)
        R('EXPR_ID(SELF_ATTR)', '0', weight=0.6)
    R('EXPRESSION(TERM, ARITH_OP, TERM)',       '0 1 2')
    R('EXPRESSION(TERM)',                        '0', weight=0.35)
    R('EXPRESSION(EXPRESSION, ARITH_OP, TERM)', '(0) 1 2', weight=0.3)
    if include_extra_ops:
        R('EXPRESSION(TERM, DIV_OP, NON_ZERO_DIGIT)', '0 1 2', weight=0.5)
    R('ENCLOSED(EXPRESSION)', '(0)')

    if include_ternary:
        R('TERNARY_INT(TERM, COND_EXPR, TERM)', '(0 if 1 else 2)')
        R('TERM(TERNARY_INT)', '0', weight=0.3)

    # Display / len / index
    R('DISP_ID(CTX)', get_last_var)
    R('DISP_EXPR(EXPR_ID, ARITH_OP, EXPR_ID)', '0 1 2')
    R('DISP_EXPR(EXPR_ID, ARITH_OP, DIGIT)',   '0 1 2')
    R('LEN_EXPR(STR_LIT)',  'len(0)')
    R('LEN_EXPR(LIST_LIT)', 'len(0)')
    R('INDEX_EXPR(LIST_LIT, SMALL_INDEX)',    '0[1]')
    R('STR_INDEX_EXPR(STR_LIT, SMALL_INDEX)', '0[1]')

    # Typed calls
    R('CALL_INT(CTX)',  lambda c: _render_call_of_type('int'))
    R('CALL_STR(CTX)',  lambda c: _render_call_of_type('str'))
    R('CALL_LIST(CTX)', lambda c: _render_call_of_type('list'))

    # String expressions
    R('STR_ATOM(STR_LIT)',  '0')
    R('STR_ATOM(CALL_STR)', '0')
    R('STR_EXPR(STR_ATOM, STR_ATOM)', '0 + 1')
    for _m in ['.upper()', '.lower()', '.strip()', '.title()']:
        R('STR_METHOD(STR_ATOM)', f'0{_m}', weight=0.5)

    if include_fstrings:
        def render_fstring(ctx):
            label = random.choice(['val', 'n', 'x', 'result', 'item', 'out'])
            return f'f"{label}={{{pv("int")}}}"'
        R('FSTR(CTX)', render_fstring)
        R('STR_ATOM(FSTR)', '0', weight=1.2)

    # List expressions
    R('LIST_ATOM(LIST_LIT)',  '0')
    R('LIST_ATOM(CALL_LIST)', '0')
    R('LIST_EXPR(LIST_ATOM, LIST_ATOM)', '0 + 1')

    if include_comprehensions:
        def render_list_comp(ctx):
            used = set().union(*S.scope.all.values()) | set(init_vals())
            lv = random.choice([c for c in chars if c not in used] or chars)
            a = random.randint(0, 5); b = a + random.randint(1, 8)
            cap = max(2, max_number // 4)
            inner = random.choice([f"{lv} + {random.randint(1, cap)}",
                                   f"{lv} * {lv}", f"{lv} % {random.randint(2, cap+1)}", lv])
            return f"[{inner} for {lv} in range({a}, {b})]"
        R('LIST_COMP(CTX)', render_list_comp)
        R('LIST_ATOM(LIST_COMP)', '0', weight=1.5)

    # Initializations
    R('INIT(CTX)', render_init)
    for n in [2, 3, 4, 5]:
        for k in range(1, n + 2):
            R(f"IDENT_INIT_{n}(" + ",".join(["INIT"] * k) + ")",
              lambda *a: "".join(x.render('py') for x in a))

    # Simple + advanced typed assignments
    R('SIMPLE_ARITH(ENCLOSED)', '0')
    R('SIMPLE_ARITH(SIMPLE_ARITH, ARITH_OP, ENCLOSED)',
      lambda *a: "".join(x.render('py') for x in a), weight=0.5)
    R('SIMPLE_ASSIGN(VAR, EXPRESSION)', render_assign)
    R('SIMPLE_ASSIGNS', '')
    R('SIMPLE_ASSIGNS(SIMPLE_ASSIGN)', '0')

    int_rhs = [('SIMPLE_ARITH', 1.0), ('EXPRESSION', 1.0), ('LEN_EXPR', 1.0),
               ('INDEX_EXPR',   1.0), ('CALL_INT',    3.0)]
    str_rhs = [('STR_LIT', 1.0), ('CALL_STR', 2.0), ('STR_EXPR', 0.8), ('STR_METHOD', 0.6)]
    lst_rhs = [('LIST_LIT', 1.0), ('CALL_LIST', 2.0), ('LIST_EXPR', 0.8)]
    if include_ternary:        int_rhs.append(('TERNARY_INT', 0.7))
    if include_fstrings:       str_rhs.append(('FSTR',        1.2))
    if include_comprehensions: lst_rhs.append(('LIST_COMP',   2.5))
    for sym, w in int_rhs: R(f'ADV_ASSIGN_TYPE(VAR, {sym})', _assign(),       weight=w)
    for sym, w in str_rhs: R(f'ADV_ASSIGN_TYPE(VAR, {sym})', _assign('str'),  weight=w)
    for sym, w in lst_rhs: R(f'ADV_ASSIGN_TYPE(VAR, {sym})', _assign('list'), weight=w)
    R('ADV_ASSIGNS', '')
    R('ADV_ASSIGNS(ADV_ASSIGN_TYPE)', '0')

    if include_augmented_assigns:
        for _op in ['+=', '-=', '*=']: R('AUG_OP', _op)
        def render_aug_assign(v_node, op_node, e_node):
            v = v_node.render('py')
            if not v.isidentifier(): return "pass\n"
            S.scope.last = {v}
            return f"{v} {op_node.render('py')} {e_node.render('py')}\n"
        R('AUG_ASSIGN(EXPR_ID, AUG_OP, TERM)', render_aug_assign)

    if include_swap:
        def render_swap(ctx):
            excl = loop_excl()
            pool = list((S.scope.safe['int'] if (safe_returns and S.nest_depth == 0)
                         else S.scope.all['int']) - excl)
            if len(pool) < 2: return "pass\n"
            a, b = random.sample(pool, 2)
            S.scope.last = {a, b}
            return f"{a}, {b} = {b}, {a}\n"
        R('SWAP_STMT(CTX)', render_swap)

    # Conditionals
    R('COND_EXPR(EXPR_ID, REL_OP, EXPR_ID)',      render_cond_expr)
    R('COND_EXPR(EXPR_ID, REL_OP, DIGIT)',         '0 1 2')
    R('COND_EXPR(LEN_EXPR, REL_OP, DIGIT)',        '0 1 2', weight=0.4)
    R('COND_EXPR(EXPR_ID, REL_OP, CALL_INT)',      '0 1 2', weight=0.4)
    R('COND_EXPR(EXPRESSION, REL_OP, EXPRESSION)', '0 1 2', weight=0.3)
    R('IF_BLK(COND_EXPR)',             'if 0:\n')
    R('IF_BLK(LOG_PREFIX, COND_EXPR)', 'if 01:\n', weight=0.3)
    R('ELIF_BLK(COND_EXPR)',           'elif 0:\n')
    R('ELSE_BLK',                      'else:\n')
    R('IF_STMT(IF_BLK, BODY_STMT)',
      lambda h, b: if_chain((h, b)))
    R('IF_STMT(IF_BLK, BODY_STMT, ELSE_BLK, BODY_STMT)',
      lambda h, b, e, s: if_chain((h, b), (e, s)))
    R('IF_STMT(IF_BLK, BODY_STMT, ELIF_BLK, BODY_STMT)',
      lambda h, b, ei, s: if_chain((h, b), (ei, s)),               weight=0.5)
    R('IF_STMT(IF_BLK, BODY_STMT, ELIF_BLK, BODY_STMT, ELSE_BLK, BODY_STMT)',
      lambda h, b, ei, s, e, t: if_chain((h, b), (ei, s), (e, t)), weight=0.4)

    # Display
    R('DISPLAY(DISP_ID)', 'print(0)')
    R('ADV_DISP(DISPLAY)',        '0')
    R('ADV_DISP(DISP_EXPR)',      'print(0)')
    R('ADV_DISP(STR_LIT)',        'print(0)')
    R('ADV_DISP(LIST_LIT)',       'print(0)')
    R('ADV_DISP(LEN_EXPR)',       'print(0)')
    R('ADV_DISP(INDEX_EXPR)',     'print(0)')
    R('ADV_DISP(STR_INDEX_EXPR)', 'print(0)')
    R('ADV_DISP(CALL_INT)',       'print(0)', weight=1.5)

    # Bare expression statements
    R('EXPR_STMT(CALL_INT)',   '0\n')
    R('EXPR_STMT(CALL_STR)',   '0\n')
    R('EXPR_STMT(CALL_LIST)',  '0\n')
    R('EXPR_STMT(EXPRESSION)', '0\n')

    # Loops
    R('FOR_INIT(CTX)',  lambda x: render_loop_math(x, 'init'))
    R('FOR_FINAL(CTX)', lambda x: render_loop_math(x, 'final_less'))
    R('STEP(CTX)',      lambda x: S.aux['loops'].get('step', '1'))
    R('FOR_HEAD(VAR, FOR_INIT, FOR_FINAL, STEP)', 'for 0 in range(1, 2, 3):')
    R('FOR_HEAD(VAR, FOR_INIT, FOR_FINAL)',       'for 0 in range(1, 2):')
    R('FOR_LOOP(FOR_HEAD, BODY_STMT)',
      lambda h, b: f"{h.render('py')}\n{_indent(render_block(b, S, loop=True))}\n")

    R('REL_LESS', '<');    R('REL_LESS', '<=')
    R('REL_GREATER', '>'); R('REL_GREATER', '>=')
    R('WHILE_VAR(CTX)',  render_while_var)
    R('WH_FINAL_L(CTX)', lambda x: render_loop_math(x, 'final_less'))
    R('WH_FINAL_G(CTX)', lambda x: render_loop_math(x, 'final_greater'))
    R('WH_UPD_L(CTX)',   lambda x: render_while_update(x, '+'))
    R('WH_UPD_G(CTX)',   lambda x: render_while_update(x, '-'))
    def _render_while(v, op, f, b, u):
        var_s  = v.render('py')
        init_s = S.aux['loops'].get('init_stmt', '')
        op_s, fin_s = op.render('py'), f.render('py')
        saved  = dict(S.aux['loops'])
        body_s = _indent(render_block(b, S, loop=True))
        S.aux['loops'].clear(); S.aux['loops'].update(saved)
        upd_s  = u.render('py')
        return f"{init_s}while {var_s} {op_s} {fin_s}:\n{body_s}\n    {upd_s}\n"
    R('WH_L(WHILE_VAR, REL_LESS,    WH_FINAL_L, BODY_STMT, WH_UPD_L)', _render_while)
    R('WH_G(WHILE_VAR, REL_GREATER, WH_FINAL_G, BODY_STMT, WH_UPD_G)', _render_while)

    # === 8. BODY_STMT — flag-gated ===
    R('BODY_STMT(SIMPLE_ASSIGN)',   '0')
    R('BODY_STMT(ADV_ASSIGN_TYPE)', '0')
    if include_augmented_assigns: R('BODY_STMT(AUG_ASSIGN)', '0', weight=0.7)
    if include_swap:              R('BODY_STMT(SWAP_STMT)',  '0', weight=0.2)
    if include_assert:
        _TAUTOLOGIES = ['True', '1 == 1', '0 < 1', 'len("hi") == 2', 'len("go") == 2',
                        '"hi" == "hi"', '2 + 2 == 4', '[0, 1, 2][0] == 0']
        R('ASSERT_STMT(COND_EXPR)',
          lambda c: f"assert {gated(failure_rate, lambda: random.choice(_TAUTOLOGIES), lambda: c.render('py'))}\n")
        R('BODY_STMT(ASSERT_STMT)', '0', weight=0.3)
    if include_print:      R('BODY_STMT(DISPLAY)',  '0')
    if include_loops:
        R('BODY_STMT(FOR_LOOP)', '0', weight=0.4)
        R('BODY_STMT(WH_L)',     '0', weight=0.3)
        R('BODY_STMT(WH_G)',     '0', weight=0.3)
    if include_conditionals: R('BODY_STMT(IF_STMT)', '0', weight=0.5)
    if include_break_continue:
        R('LOOP_CTL_STMT(CTX)',
          lambda x: (random.choice(['break', 'continue']) if S.loop_depth > 0 else 'pass') + '\n')
        R('BODY_STMT(LOOP_CTL_STMT)', '0', weight=0.2)
    if include_try_except:
        R('TRY_STMT(BODY_STMT, BODY_STMT)',
          lambda b, h: f"try:\n{_indent(render_block(b, S))}\nexcept Exception:\n{_indent(render_block(h, S))}\n")
        R('BODY_STMT(TRY_STMT)', '0', weight=0.25)

    # STMTS: recursive nonterminal — depth scales body length.
    def _norm(text): return text.rstrip('\n') + '\n'
    R('STMTS(BODY_STMT)', lambda b: _norm(b.render('py')))
    R('STMTS(STMTS, BODY_STMT)',
      lambda s, b: _norm(s.render('py')) + _norm(b.render('py')), weight=0.6)
    if min_body_stmts >= 2:
        R('STMTS(BODY_STMT, BODY_STMT)',
          lambda a, b: _norm(a.render('py')) + _norm(b.render('py')))
    R('FUNC_DEF(STMTS)', render_func_def)

    if include_classes:
        R('METHOD_DEF(STMTS)', render_method_def)
        R('METHODS(METHOD_DEF)',          lambda m: m.render('py'))
        R('METHODS(METHODS, METHOD_DEF)', lambda ms, m: ms.render('py') + m.render('py'), weight=0.5)
        R('CLASS_DEF(METHODS)', render_class_def)
        R('INSTANCE_USE(CTX)', render_instance_use)

    # === 9. TOP_STMT / FINAL_STMT ===
    _stmt_end = lambda c: c.render('py').rstrip('\n') + '\n'
    R('TOP_ASSIGNS(ADV_ASSIGNS)', _stmt_end)
    R('TOP_DISP(ADV_DISP)',       _stmt_end)
    R('TOP_FOR(FOR_LOOP)',        _stmt_end)
    R('TOP_WH_L(WH_L)',           _stmt_end)
    R('TOP_WH_G(WH_G)',           _stmt_end)
    R('TOP_IF(IF_STMT)',          _stmt_end)
    R('TOP_EXPR(EXPR_STMT)',      _stmt_end)
    R('TOP_STMT(TOP_ASSIGNS)', '0')
    if include_print:      R('TOP_STMT(TOP_DISP)',  '0', weight=1.2)
    if include_loops:
        R('TOP_STMT(TOP_FOR)',  '0', weight=0.5)
        R('TOP_STMT(TOP_WH_L)', '0', weight=0.35)
        R('TOP_STMT(TOP_WH_G)', '0', weight=0.35)
    if include_conditionals: R('TOP_STMT(TOP_IF)', '0', weight=0.6)
    R('TOP_STMT(TOP_EXPR)', '0')

    def render_main_call(ctx):
        spec = S.defs.get('f0')
        if spec is None: return get_atom(ctx)
        return f"f0({', '.join(_arg_of_type(t) for t in spec['ptypes'])})"
    R('MAIN_CALL(CTX)', render_main_call)
    if n_functions > 0:
        R('FINAL_STMT(MAIN_CALL)',
          (lambda c: f"print({c.render('py')})\n") if include_print
          else (lambda c: f"{c.render('py')}\n"))
    elif include_print:
        R('FINAL_STMT(TOP_DISP)', '0')
    else:
        R('FINAL_STMT(TOP_EXPR)', '0')

    # === 10. Entry point ===
    if mode == 'function':
        prog_args = ['FUNC_DEF'] * max(1, n_functions)
        if include_classes and n_classes > 0:
            prog_args = prog_args + ['CLASS_DEF'] * n_classes + ['INSTANCE_USE']
    else:
        needs_fb = n_functions > 0 and 'int' not in param_types
        eff_inits = max(n_outer_inits, 2 if needs_fb else n_outer_inits)
        init_name = f'IDENT_INIT_{max(2, eff_inits)}' if eff_inits > 0 else None
        prog_args = ([init_name] if init_name else []) \
                    + ['FUNC_DEF'] * max(0, n_functions) + ['TOP_STMT', 'FINAL_STMT']
    R(f"PROGRAM({','.join(prog_args)})", lambda *a: "".join(x.render('py') for x in a))
    R('ALL(RESET, PROGRAM)', lambda r, p: r.render('py') + p.render('py'))
    R('start(ALL)', '0')
    return R
