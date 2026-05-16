import random
from .. import Substitution, Constraint, generate, init_grammar

# Types: 'int', 'str', 'list'. state['funcs'][name] = (arity, ret_t, param_types).
# Typed call rules (CALL_INT/STR/LIST) filter by return type, fall back to a literal.
# Safety pyramid: scope['vars'][t] (any assigned) ⊇ scope['safe_vars'][t] (definitely
# assigned at nest_depth==0). safe_returns=True restricts returns/refs to safe_vars.
# Depth scaling: STMTS / EXPRESSION are recursive — body length and arithmetic depth
# grow with max_depth. include_* flags compose orthogonally.


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
    max_params=2,               # max parameters per function (1..max_params)
    # --- call graph ----------------------------
    allow_recursion=True,
    allow_cross_calls=True,
    # --- semantic quality ----------------------
    safe_returns=True,          # return only definitely-assigned vars (prevents UnboundLocalError)
    min_body_stmts=1,           # minimum statements in a function body (1 or 2)
    failure_rate=0.5,           # 0=avoid runtime failures (asserts tautological, no recur-in-return,
                                # no '//' '%'); 1=allow all risky variants. Multiplies with include_*.
    triviality_rate=0.5,        # 0=reject bare-var/literal returns when richer options exist;
                                # 1=accept any return form. Affects return-value selection only.
    # --- feature flags -------------------------
    include_print=True,
    include_loops=True,
    include_conditionals=True,
    include_augmented_assigns=True,   # x += y, x -= y, x *= y
    include_ternary=True,             # (a if cond else b)
    include_assert=True,              # assert cond
    include_comprehensions=True,      # [expr for v in range(a, b)]
    include_fstrings=True,            # f"label={x}"
    include_extra_ops=True,           # // and % (may cause ZeroDivisionError at runtime)
    include_swap=True,                # a, b = b, a
    include_break_continue=True,      # break/continue inside loops
    include_try_except=True,          # try/except blocks
):
    R = init_grammar(['py'])
    chars = list("abcdefghijklmnopqrstuvwxyz")
    param_types = tuple(param_types)
    return_types = tuple(return_types) if return_types is not None else param_types
    if mode not in ('program', 'function'):
        raise ValueError(f"mode must be 'program' or 'function', got {mode!r}")
    if f0_is_root is None:
        f0_is_root = (mode == 'function')
    if mode == 'function' and n_functions < 1:
        raise ValueError("mode='function' requires n_functions >= 1")

    # === 1. Scope stack + type registry ===
    # nest_depth tracks how many conditional/loop blocks deep we are.
    # Variables assigned at nest_depth == 0 are "definitely assigned" (safe).
    state = {'scopes': [], 'loops': {}, 'funcs': {}, 'current_fn': None,
             'nest_depth': 0, 'loop_depth': 0}
    TYPES = ('int', 'str', 'list')
    _TYPE_LITERALS = {
        'int':  lambda: str(random.randint(0, max_number)),
        'str':  lambda: random.choice(['"hi"', '"cat"', '"go"', '"sun"']),
        'list': lambda: "[0, 1, 2]",
    }

    def _new_scope(params=None, p_types=None):
        s = {'assigned': {}, 'last': set(), 'params': set(),
             'vars': {t: set() for t in TYPES},
             'safe_vars': {t: set() for t in TYPES}}  # definitely-assigned vars
        for p, t in zip(params or (), p_types or ()):
            s['assigned'][p] = '0'
            s['vars'][t].add(p)
            s['safe_vars'][t].add(p)   # params are always safe
            s['params'].add(p)
        return s

    def cur(): return state['scopes'][-1]
    def push_scope(params=None, p_types=None):
        state['scopes'].append(_new_scope(params, p_types))
    def pop_scope(): state['scopes'].pop()

    def reset_state(ctx_node):
        state.update({'scopes': [_new_scope()], 'loops': {}, 'funcs': {},
                      'current_fn': None, 'fn_plan': [], 'fn_plan_idx': 0,
                      'nest_depth': 0, 'loop_depth': 0})
        if n_functions > 0:
            _make_plan()
        return ""

    # === 2. Context-sensitive helpers ===
    def concat(*args): return "".join(a.render('py') for a in args)

    def render_init(ctx_node):
        top = cur()
        pool = [c for c in chars if c not in top['assigned']]
        v = random.choice(pool or chars)
        d = str(random.randint(0, max_number))
        top['assigned'][v] = d
        top['vars']['int'].add(v)
        top['safe_vars']['int'].add(v)   # INIT is always at function entry level
        return f"{v} = {d}\n"

    def render_assign(v_node, e_node, kind='int'):
        v, e = v_node.render('py'), e_node.render('py')
        top = cur()
        top['last'] = {v}
        top['assigned'].setdefault(v, '0')
        for s in top['vars'].values(): s.discard(v)
        top['vars'][kind].add(v)
        if state['nest_depth'] == 0:        # only safe if at top level of function body
            for s in top['safe_vars'].values(): s.discard(v)
            top['safe_vars'][kind].add(v)
        return f"{v} = {e}\n"

    def _loop_protected_vars():
        # While-loop var must not be re-assigned in its body (would break convergence).
        v = state['loops'].get('var') if state['nest_depth'] > 0 else None
        return {v} if v else set()

    def _pick_var(t, *, create=False):
        # Walk scope chain for a `t`-typed var. `create=True` fabricates a fresh
        # name for an assignment target; `create=False` falls back to a literal
        # (for expression contexts — avoids phantom vars → UnboundLocalError).
        use_safe = safe_returns and state['nest_depth'] == 0
        excl = _loop_protected_vars()
        for scope in reversed(state['scopes']):
            pool = (scope['safe_vars'][t] if use_safe else scope['vars'][t]) - excl
            if pool: return random.choice(list(pool))
        if use_safe:
            for scope in reversed(state['scopes']):
                pool = scope['vars'][t] - excl
                if pool: return random.choice(list(pool))
        if not create: return _TYPE_LITERALS[t]()
        outer = state['scopes'][0]
        v = (next((c for c in chars if c not in outer['assigned'] and c not in excl), None)
             or next((c for c in chars if c not in excl), random.choice(chars)))
        outer['assigned'][v] = '0'   # name reserved; render_assign registers the var
        return v

    def get_var_of_type(t):    return _pick_var(t, create=True)   # assignment target
    def get_expr_var(t='int'): return _pick_var(t)                # expression context
    def get_assigned_var(ctx): return _pick_var('int')
    def get_atom(ctx):
        if cur()['assigned'] and random.random() < 0.7:
            return _pick_var('int')
        return str(random.randint(0, max_number))
    def get_last_var(ctx):
        top = cur()
        if top['last'] and not (safe_returns and state['nest_depth'] == 0):
            return next(iter(top['last']))
        for scope in reversed(state['scopes']):
            for t in TYPES:
                pool = scope['safe_vars'][t] - _loop_protected_vars()
                if pool: return random.choice(list(pool))
        return _pick_var('int')

    def render_loop_math(ctx_node, mode):
        if mode == 'init':
            val = str(random.randint(0, 20))
            state['loops']['val'] = val
            return val
        init_val = int(state['loops'].get('val', '0'))
        step, count = random.choice([(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)])
        state['loops']['step'] = str(step)
        if mode == 'final_less':    return str(step * count + init_val - 1)
        if mode == 'final_greater': return str(init_val - step * count + 1)
        return "0"

    def render_while_var(ctx_node):
        # The while-loop var must be definitely assigned before the loop starts.
        excl = _loop_protected_vars()
        v = None
        for scope in reversed(state['scopes']):
            pool = scope['safe_vars']['int'] - excl
            if pool:
                v = random.choice(list(pool))
                state['loops']['init_stmt'] = ''
                break
        if v is None:
            # No safe int var available: create one with an init statement.
            v = next((c for c in chars if c not in cur()['assigned'] and c not in excl), None)
            if v is None:
                v = next((c for c in chars if c not in excl), random.choice(chars))
            cur()['assigned'][v] = '0'
            cur()['vars']['int'].add(v)
            cur()['safe_vars']['int'].add(v)
            state['loops']['init_stmt'] = f"{v} = 0\n"
        state['loops']['var'] = v
        state['loops']['val'] = cur()['assigned'].get(v, '0')
        return v

    def render_while_update(ctx_node, op):
        v, s = state['loops'].get('var', 'i'), state['loops'].get('step', '1')
        return f"{v} = {v} {op} {s}"

    def render_cond_expr(v1_node, op_node, v2_node):
        lhs, op, rhs = v1_node.render('py'), op_node.render('py'), v2_node.render('py')
        if lhs == rhs and random.random() < 0.8:
            scope = cur()
            pool = scope['safe_vars']['int'] if (safe_returns and state['nest_depth'] == 0) else scope['vars']['int']
            alts = [c for c in pool if c != lhs]
            if alts: rhs = random.choice(alts)
        return f"{lhs} {op} {rhs}"

    # === 3. Structural helpers ===
    def _indent_block(text, prefix='    '):
        out = []
        for line in text.split('\n'):
            if not line.strip(): continue
            out.append(prefix + line.replace('\t', '    '))
        return '\n'.join(out)

    def _in_block(node, is_loop=False):  # render inside one nest level
        state['nest_depth'] += 1
        if is_loop: state['loop_depth'] += 1
        text = node.render('py')
        state['nest_depth'] -= 1
        if is_loop: state['loop_depth'] -= 1
        return text

    def _if_chain(*pairs):  # (head_node, body_node) pairs → if/elif/else chain
        return ''.join(f"{h.render('py')}{_indent_block(_in_block(b))}\n" for h, b in pairs)

    def _assign(kind='int'):
        return render_assign if kind == 'int' else lambda v, e: render_assign(v, e, kind=kind)

    # === 4. Typed calls ===
    def _fn_index(name):
        return int(name[1:]) if name and name.startswith('f') else -1

    def _callable_functions(ret_type):
        # Self-recursion outside a conditional/loop body is gated by failure_rate
        # (unguarded self-calls almost always RecursionError; inside nest_depth>0
        # there's at least a chance the path is taken conditionally).
        candidates = []
        current = state['current_fn']
        current_idx = _fn_index(current)
        unguarded_self = state['nest_depth'] == 0
        for fname, (arity, rt, ptypes) in state['funcs'].items():
            if rt != ret_type: continue
            is_self = (fname == current)
            if is_self and not allow_recursion: continue
            if is_self and unguarded_self and random.random() >= failure_rate: continue
            if (not is_self) and current is not None and not allow_cross_calls: continue
            if f0_is_root and current is not None and _fn_index(fname) < current_idx: continue
            candidates.append((fname, ptypes))
        return candidates

    def _render_call_of_type(ret_type):
        cands = _callable_functions(ret_type)
        if cands:
            fname, ptypes = random.choice(cands)
            return f"{fname}({', '.join(_arg_of_type(t) for t in ptypes)})"
        return _TYPE_LITERALS[ret_type]()

    def _arg_of_type(t):
        use_safe = safe_returns and state['nest_depth'] == 0
        for scope in reversed(state['scopes']):
            pool = scope['safe_vars'][t] if use_safe else scope['vars'][t]
            if pool and random.random() < 0.6:
                return random.choice(list(pool))
        if use_safe:  # fallback to full pool if safe pool is empty
            for scope in reversed(state['scopes']):
                if scope['vars'][t] and random.random() < 0.6:
                    return random.choice(list(scope['vars'][t]))
        return _TYPE_LITERALS[t]()

    render_call_int  = lambda ctx: _render_call_of_type('int')
    render_call_str  = lambda ctx: _render_call_of_type('str')
    render_call_list = lambda ctx: _render_call_of_type('list')

    # === 5. Function defs ===
    def _pick_param_types(n):
        return [random.choice(list(param_types)) for _ in range(n)]

    def _make_plan():
        plan = []
        for i in range(n_functions):
            if i == 0 and main_signature is not None:
                ptypes_i, ret_t_i = main_signature
                ptypes_i = list(ptypes_i)
                n_params = len(ptypes_i)
            else:
                n_params = random.randint(1, max(1, max_params))
                ptypes_i = _pick_param_types(n_params)
                ret_t_i = random.choice(list(return_types))
            fname = f"f{i}"
            state['funcs'][fname] = (n_params, ret_t_i, ptypes_i)
            plan.append((fname, n_params, ret_t_i, ptypes_i))
        state['fn_plan'] = plan
        state['fn_plan_idx'] = 0

    _DEFAULT_RET = {'int': '0', 'str': '""', 'list': '[]'}
    def _pick_return_value(ret_t):
        # Tier 1: call/arith/ternary forms (only when safe_pool has ≥2 ints).
        # Self-recursion in return is gated by failure_rate: it usually leads to
        # RecursionError unless the body has a base case (which CFG can't enforce).
        top = cur()
        safe_pool = top['safe_vars'][ret_t]
        recur_ok = allow_recursion and random.random() < failure_rate
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
        # Tier 2: pick a safe variable, then params-only, then literal.
        pool = safe_pool if safe_returns else top['vars'][ret_t]
        # triviality_rate: when low, avoid bare param/var/literal returns by
        # synthesizing a compound expression (arith with another var, or arith
        # with a literal when only one var is available).
        if pool and ret_t == 'int' and random.random() > triviality_rate:
            if len(pool) >= 2:
                a, b = random.sample(list(pool), 2)
                return f"{a} {random.choice(['+', '-', '*'])} {b}"
            a = next(iter(pool))
            return f"{a} {random.choice(['+', '-', '*'])} {random.randint(1, max_number)}"
        if top['last'] and (last := [v for v in top['last'] if v in pool]):
            return last[0]
        if pool:
            return random.choice(list(pool))
        if safe_returns and (param_pool := top['vars'][ret_t] & top['params']):
            return random.choice(list(param_pool))
        return _DEFAULT_RET[ret_t]

    def render_func_def(body_node):
        idx = state['fn_plan_idx']
        state['fn_plan_idx'] = idx + 1
        fname, n_params, ret_t, ptypes = state['fn_plan'][idx]

        outer = cur()['assigned']
        pool = [c for c in chars if c not in outer]
        params = (random.sample(pool, n_params) if len(pool) >= n_params
                  else random.sample(chars, n_params))

        prev_fn   = state['current_fn']
        prev_nest = state['nest_depth']
        state['current_fn'] = fname
        state['nest_depth'] = 0
        push_scope(params=params, p_types=ptypes)
        body_text = body_node.render('py')

        ret_v = _pick_return_value(ret_t)
        pop_scope()
        state['current_fn'] = prev_fn
        state['nest_depth']  = prev_nest

        if type_hints:
            sig = ', '.join(f"{p}: {t}" for p, t in zip(params, ptypes))
            header = f"def {fname}({sig}) -> {ret_t}:" if returns else f"def {fname}({sig}):"
        else:
            header = f"def {fname}({', '.join(params)}):"

        indented = _indent_block(body_text)
        tail = f"\n    return {ret_v}" if returns else ("" if indented else "\n    pass")
        return f"{header}\n{indented}{tail}\n"

    # === 6. Grammar ===
    R('CTX', '')
    R('RESET(CTX)', reset_state)

    # Terminals
    R('DIGIT',       lambda: str(random.randint(0, max_number)))
    R('SMALL_INDEX', lambda: str(random.randint(0, 1)))
    R('STR_LIT',     lambda: random.choice(['"hi"', '"cat"', '"go"', '"sun"']))
    R('LIST_LIT(DIGIT, DIGIT, DIGIT)', '[0, 1, 2]')
    R('VAR(CTX)',    lambda x: get_var_of_type('int'))
    R('BOOL_LIT',    lambda: random.choice(['True', 'False']))

    for _op in ['+', '-', '*']:
        R('ARITH_OP', _op)
    for _op in ['<', '>', '<=', '>=', '!=', '==']: R('REL_OP',    _op)
    for _op in ['and', 'or']:                       R('LOG_INFIX', _op)
    R('LOG_PREFIX', 'not ')

    # Non-zero denominator terminal, used by // and % to prevent ZeroDivisionError.
    R('NON_ZERO_DIGIT', lambda: str(random.randint(1, max(1, max_number))))
    if include_extra_ops:
        R('DIV_OP', '//')
        R('DIV_OP', '%')

    # Integer expressions — EXPRESSION is recursive for depth scaling.
    R('EXPR_ID(CTX)', get_assigned_var)
    R('ATOM(CTX)',    get_atom)
    R('ATOM(BOOL_LIT)', '0', weight=0.15)
    R('TERM(EXPR_ID)', '0'); R('TERM(DIGIT)', '0'); R('TERM(ATOM)', '0')
    R('TERM(CALL_INT)', '0', weight=0.6)
    R('EXPRESSION(TERM, ARITH_OP, TERM)',       '0 1 2')
    R('EXPRESSION(TERM)',                        '0',         weight=0.35)
    R('EXPRESSION(EXPRESSION, ARITH_OP, TERM)', '(0) 1 2',   weight=0.3)
    if include_extra_ops:
        R('EXPRESSION(TERM, DIV_OP, NON_ZERO_DIGIT)', '0 1 2', weight=0.5)
    R('ENCLOSED(EXPRESSION)', '(0)')

    if include_ternary:
        R('TERNARY_INT(TERM, COND_EXPR, TERM)', '(0 if 1 else 2)')
        R('TERM(TERNARY_INT)', '0', weight=0.3)

    # Display expressions
    R('DISP_ID(CTX)', get_last_var)
    R('DISP_EXPR(EXPR_ID, ARITH_OP, EXPR_ID)', '0 1 2')
    R('DISP_EXPR(EXPR_ID, ARITH_OP, DIGIT)',   '0 1 2')
    R('LEN_EXPR(STR_LIT)',  'len(0)')
    R('LEN_EXPR(LIST_LIT)', 'len(0)')
    R('INDEX_EXPR(LIST_LIT, SMALL_INDEX)',    '0[1]')
    R('STR_INDEX_EXPR(STR_LIT, SMALL_INDEX)', '0[1]')

    # Typed call rules
    R('CALL_INT(CTX)',  render_call_int)
    R('CALL_STR(CTX)',  render_call_str)
    R('CALL_LIST(CTX)', render_call_list)

    # String expressions
    R('STR_ATOM(STR_LIT)',  '0')
    R('STR_ATOM(CALL_STR)', '0')
    R('STR_EXPR(STR_ATOM, STR_ATOM)', '0 + 1')
    for _m in ['.upper()', '.lower()', '.strip()', '.title()']:
        R('STR_METHOD(STR_ATOM)', f'0{_m}', weight=0.5)

    if include_fstrings:
        def render_fstring(ctx):
            var = get_expr_var('int')
            label = random.choice(['val', 'n', 'x', 'result', 'item', 'out'])
            return f'f"{label}={{{var}}}"'
        R('FSTR(CTX)', render_fstring)
        R('STR_ATOM(FSTR)', '0', weight=1.2)

    # List expressions
    R('LIST_ATOM(LIST_LIT)',  '0')
    R('LIST_ATOM(CALL_LIST)', '0')
    R('LIST_EXPR(LIST_ATOM, LIST_ATOM)', '0 + 1')

    if include_comprehensions:
        def render_list_comp(ctx):
            scope = cur()
            used = set(scope['assigned'].keys())
            lv = random.choice([c for c in chars if c not in used] or chars)
            a = random.randint(0, 5)
            b = a + random.randint(1, 8)
            cap = max(2, max_number // 4)
            inner = random.choice([
                f"{lv} + {random.randint(1, cap)}",
                f"{lv} * {lv}",
                f"{lv} % {random.randint(2, cap + 1)}",
                lv,
            ])
            return f"[{inner} for {lv} in range({a}, {b})]"
        R('LIST_COMP(CTX)', render_list_comp)
        R('LIST_ATOM(LIST_COMP)', '0', weight=1.5)

    # Initializations
    R('INIT(CTX)', render_init)
    for n in [2, 3, 4, 5]:
        for k in range(1, n + 2):
            R(f"IDENT_INIT_{n}(" + ",".join(["INIT"] * k) + ")", concat)

    # Simple assignments
    R('SIMPLE_ARITH(ENCLOSED)', '0')
    R('SIMPLE_ARITH(SIMPLE_ARITH, ARITH_OP, ENCLOSED)', concat, weight=0.5)
    R('SIMPLE_ASSIGN(VAR, EXPRESSION)', render_assign)
    R('SIMPLE_ASSIGNS', '')
    R('SIMPLE_ASSIGNS(SIMPLE_ASSIGN)', '0')

    # Typed assignments — _assign() removes repeated lambda-per-kind.
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

    # Augmented assignments
    if include_augmented_assigns:
        for _op in ['+=', '-=', '*=']: R('AUG_OP', _op)
        def render_aug_assign(v_node, op_node, e_node):
            v = v_node.render('py')
            # If EXPR_ID fell back to a literal (no suitable var), skip the aug assign.
            if not v.isidentifier():
                return f"pass\n"
            cur()['last'] = {v}
            return f"{v} {op_node.render('py')} {e_node.render('py')}\n"
        R('AUG_ASSIGN(EXPR_ID, AUG_OP, TERM)', render_aug_assign)

    # Variable swap
    if include_swap:
        def render_swap(ctx):
            scope = cur()
            excl = _loop_protected_vars()
            pool = list((scope['safe_vars']['int'] if (safe_returns and state['nest_depth'] == 0) else scope['vars']['int']) - excl)
            if len(pool) < 2:
                return "pass\n"
            a, b = random.sample(pool, 2)
            scope['last'] = {a, b}
            return f"{a}, {b} = {b}, {a}\n"
        R('SWAP_STMT(CTX)', render_swap)

    # Conditionals — richer conditions via EXPRESSION at depth.
    R('COND_EXPR(EXPR_ID, REL_OP, EXPR_ID)',      render_cond_expr)
    R('COND_EXPR(EXPR_ID, REL_OP, DIGIT)',         '0 1 2')
    R('COND_EXPR(LEN_EXPR, REL_OP, DIGIT)',        '0 1 2', weight=0.4)
    R('COND_EXPR(EXPR_ID, REL_OP, CALL_INT)',      '0 1 2', weight=0.4)
    R('COND_EXPR(EXPRESSION, REL_OP, EXPRESSION)', '0 1 2', weight=0.3)
    R('IF_BLK(COND_EXPR)',             'if 0:\n')
    R('IF_BLK(LOG_PREFIX, COND_EXPR)', 'if 01:\n', weight=0.3)
    R('ELIF_BLK(COND_EXPR)',           'elif 0:\n')
    R('ELSE_BLK',                      'else:\n')

    # if/elif/else — _if_chain() / _in_block() handle indentation + nest tracking.
    R('IF_STMT(IF_BLK, BODY_STMT)',
      lambda h, b: _if_chain((h, b)))
    R('IF_STMT(IF_BLK, BODY_STMT, ELSE_BLK, BODY_STMT)',
      lambda h, b, e, s: _if_chain((h, b), (e, s)))
    R('IF_STMT(IF_BLK, BODY_STMT, ELIF_BLK, BODY_STMT)',
      lambda h, b, ei, s: _if_chain((h, b), (ei, s)),               weight=0.5)
    R('IF_STMT(IF_BLK, BODY_STMT, ELIF_BLK, BODY_STMT, ELSE_BLK, BODY_STMT)',
      lambda h, b, ei, s, e, t: _if_chain((h, b), (ei, s), (e, t)), weight=0.4)

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

    # Loops — _in_block() ensures body vars are marked as nest-level > 0.
    R('FOR_INIT(CTX)',  lambda x: render_loop_math(x, 'init'))
    R('FOR_FINAL(CTX)', lambda x: render_loop_math(x, 'final_less'))
    R('STEP(CTX)',      lambda x: state['loops'].get('step', '1'))
    R('FOR_HEAD(VAR, FOR_INIT, FOR_FINAL, STEP)', 'for 0 in range(1, 2, 3):')
    R('FOR_HEAD(VAR, FOR_INIT, FOR_FINAL)',       'for 0 in range(1, 2):')
    R('FOR_LOOP(FOR_HEAD, BODY_STMT)',
      lambda h, b: f"{h.render('py')}\n{_indent_block(_in_block(b, is_loop=True))}\n")

    R('REL_LESS', '<');    R('REL_LESS', '<=')
    R('REL_GREATER', '>'); R('REL_GREATER', '>=')
    R('WHILE_VAR(CTX)',  render_while_var)
    R('WH_FINAL_L(CTX)', lambda x: render_loop_math(x, 'final_less'))
    R('WH_FINAL_G(CTX)', lambda x: render_loop_math(x, 'final_greater'))
    R('WH_UPD_L(CTX)',   lambda x: render_while_update(x, '+'))
    R('WH_UPD_G(CTX)',   lambda x: render_while_update(x, '-'))
    def _render_while(v, op, f, b, u):
        var_s  = v.render('py')
        init_s = state['loops'].get('init_stmt', '')
        op_s   = op.render('py')
        fin_s  = f.render('py')
        saved  = dict(state['loops'])
        body_s = _indent_block(_in_block(b, is_loop=True))
        state['loops'].clear(); state['loops'].update(saved)
        upd_s  = u.render('py')
        return f"{init_s}while {var_s} {op_s} {fin_s}:\n{body_s}\n    {upd_s}\n"
    R('WH_L(WHILE_VAR, REL_LESS,    WH_FINAL_L, BODY_STMT, WH_UPD_L)', _render_while)
    R('WH_G(WHILE_VAR, REL_GREATER, WH_FINAL_G, BODY_STMT, WH_UPD_G)', _render_while)

    # === 7. BODY_STMT — flag-gated ===
    R('BODY_STMT(SIMPLE_ASSIGN)',   '0')
    R('BODY_STMT(ADV_ASSIGN_TYPE)', '0')
    if include_augmented_assigns: R('BODY_STMT(AUG_ASSIGN)', '0', weight=0.7)
    if include_swap:              R('BODY_STMT(SWAP_STMT)',  '0', weight=0.2)
    if include_assert:
        _TAUTOLOGIES = ['True', '1 == 1', '0 < 1', 'len("hi") == 2', 'len("go") == 2',
                        '"hi" == "hi"', '2 + 2 == 4', '[0, 1, 2][0] == 0']
        def render_assert(cond_node):
            cond = cond_node.render('py') if random.random() < failure_rate else random.choice(_TAUTOLOGIES)
            return f"assert {cond}\n"
        R('ASSERT_STMT(COND_EXPR)', render_assert)
        R('BODY_STMT(ASSERT_STMT)', '0', weight=0.3)
    if include_print:      R('BODY_STMT(DISPLAY)',  '0')
    if include_loops:
        R('BODY_STMT(FOR_LOOP)', '0', weight=0.4)
        R('BODY_STMT(WH_L)',     '0', weight=0.3)
        R('BODY_STMT(WH_G)',     '0', weight=0.3)
    if include_conditionals: R('BODY_STMT(IF_STMT)', '0', weight=0.5)
    if include_break_continue:
        # break/continue render to 'pass' outside a loop body (avoids SyntaxError).
        R('LOOP_CTL_STMT(CTX)',
          lambda x: (random.choice(['break', 'continue']) if state['loop_depth'] > 0 else 'pass') + '\n')
        R('BODY_STMT(LOOP_CTL_STMT)', '0', weight=0.2)
    if include_try_except:
        # try: <body> except Exception: <handler>. _in_block tracks nest_depth so
        # vars assigned inside try/except don't enter safe_vars (may be skipped).
        R('TRY_STMT(BODY_STMT, BODY_STMT)',
          lambda b, h: f"try:\n{_indent_block(_in_block(b))}\nexcept Exception:\n{_indent_block(_in_block(h))}\n")
        R('BODY_STMT(TRY_STMT)', '0', weight=0.25)

    # STMTS: recursive nonterminal — depth-scales body length naturally.
    # _norm() ensures every statement ends with exactly one newline.
    def _norm(text): return text.rstrip('\n') + '\n'
    R('STMTS(BODY_STMT)', lambda b: _norm(b.render('py')))
    R('STMTS(STMTS, BODY_STMT)',
      lambda s, b: _norm(s.render('py')) + _norm(b.render('py')), weight=0.6)
    # min_body_stmts=2 forces at least two statements as the base case.
    if min_body_stmts >= 2:
        R('STMTS(BODY_STMT, BODY_STMT)',
          lambda a, b: _norm(a.render('py')) + _norm(b.render('py')))
    R('FUNC_DEF(STMTS)', render_func_def)

    # === 8. TOP_STMT / FINAL_STMT — flag-gated ===
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
        spec = state['funcs'].get('f0')
        if spec is None: return get_atom(ctx)
        _, _, ptypes = spec
        return f"f0({', '.join(_arg_of_type(t) for t in ptypes)})"
    R('MAIN_CALL(CTX)', render_main_call)

    if n_functions > 0:
        if include_print:
            R('FINAL_STMT(MAIN_CALL)', lambda c: f"print({c.render('py')})\n")
        else:
            R('FINAL_STMT(MAIN_CALL)', lambda c: f"{c.render('py')}\n")
    elif include_print:
        R('FINAL_STMT(TOP_DISP)', '0')
    else:
        R('FINAL_STMT(TOP_EXPR)', '0')

    # === 9. Entry point ===
    if mode == 'function':
        prog_args = ['FUNC_DEF'] * max(1, n_functions)
    else:
        needs_fallback = n_functions > 0 and 'int' not in param_types
        effective_inits = max(n_outer_inits, 2 if needs_fallback else n_outer_inits)
        init_name = f'IDENT_INIT_{max(2, effective_inits)}' if effective_inits > 0 else None
        prog_args = []
        if init_name: prog_args.append(init_name)
        prog_args.extend(['FUNC_DEF'] * max(0, n_functions))
        prog_args.extend(['TOP_STMT', 'FINAL_STMT'])
    R(f"PROGRAM({','.join(prog_args)})", concat)

    R('ALL(RESET, PROGRAM)', lambda r, p: r.render('py') + p.render('py'))
    R('start(ALL)', '0')
    return R
