import random
from .. import Substitution, Constraint, generate, init_grammar

def tinypy_grammar(level=None, max_number=16):
    R = init_grammar(['py'])

    # -------------------------------------------------------------------------
    # 1. State Store & Reset Logic
    # -------------------------------------------------------------------------
    # This dictionary persists across generation calls, so we must reset it manually.
    state = {'assigned': {}, 'last': set(), 'loops': {}, 'list_vars': set(), 'str_vars': set(), 'numeric_vars': set()}
    chars = list("abcdefghijklmnopqrstuvwxyz")

    def reset_state(ctx_node):
        """Executed at the start of every program render to clean state."""
        state['assigned'].clear()
        state['last'].clear()
        state['loops'].clear()
        state['list_vars'].clear()
        state['str_vars'].clear()
        state['numeric_vars'].clear()
        return "" # Render nothing, just side-effect

    # -------------------------------------------------------------------------
    # 2. Context-Sensitive Helpers
    # -------------------------------------------------------------------------
    def concat(*args):
        """Concatenates rendered children."""
        return "".join(a.render('py') for a in args)

    def render_init(ctx_node):
        """Creates a new variable assignment."""
        # Prefer unused variables
        candidates = [c for c in chars if c not in state['assigned']]
        v = random.choice(candidates if candidates else chars)
        d = str(random.randint(0, max_number))
        state['assigned'][v] = d
        state['numeric_vars'].add(v)
        return f"{v} = {d}\n"

    def render_assign(v_node, e_node, kind='num'):
        """Updates 'last' variable tracking on assignment."""
        v, e = v_node.render('py'), e_node.render('py')
        state['last'].clear()
        state['last'].add(v)
        # Ensure it's marked as assigned
        state['assigned'][v] = state['assigned'].get(v, '0')
        state['list_vars'].discard(v)
        state['str_vars'].discard(v)
        state['numeric_vars'].discard(v)
        if kind == 'list':
            state['list_vars'].add(v)
        elif kind == 'str':
            state['str_vars'].add(v)
        else:
            state['numeric_vars'].add(v)
        return f"{v} = {e}\n"

    def get_assigned_var(ctx):
        """Returns a known variable name, creating one if state is empty."""
        if state['numeric_vars']:
            return random.choice(list(state['numeric_vars']))
        if state['assigned']:
            return random.choice(list(state['assigned'].keys()))
        v = random.choice(chars)
        state['assigned'][v] = '0'
        state['numeric_vars'].add(v)
        return v

    def get_atom(ctx):
        """Returns an arithmetic atom: a variable or a literal."""
        if state['assigned'] and random.random() < 0.7:
            return get_assigned_var(ctx)
        return str(random.randint(0, max_number))

    def get_last_var(ctx):
        """Returns the most recently modified variable (for print)."""
        if state['last']: return list(state['last'])[0]
        return get_assigned_var(ctx)

    def render_loop_math(ctx_node, mode):
        """Calculates Loop Bounds (Init/Step/Final) to ensure math coherence."""
        if mode == 'init':
            val = str(random.randint(0, 20))
            state['loops']['val'] = val
            return val
        
        init_val = int(state['loops'].get('val', '0'))
        # TinyPy specific step/count distribution
        step, count = random.choice([(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)])
        state['loops']['step'] = str(step)
        
        if mode == 'final_less': return str(step * count + init_val - 1)
        if mode == 'final_greater': return str(init_val - step * count + 1)
        return "0"

    def render_while_var(ctx_node):
        """Selects a variable for a while loop and caches it."""
        v = get_assigned_var(ctx_node)
        state['loops']['var'] = v
        state['loops']['val'] = state['assigned'].get(v, '0')
        return v

    def render_while_update(ctx_node, op):
        """Constructs update statement matching the loop variable."""
        v = state['loops'].get('var', 'i')
        s = state['loops'].get('step', '1')
        return f"{v} = {v} {op} {s}"

    def render_cond_expr(v1_node, op_node, v2_node):
        """Render var-vs-var conditions while limiting degenerate tautologies."""
        lhs = v1_node.render('py')
        op = op_node.render('py')
        rhs = v2_node.render('py')

        # Keep self-comparisons occasionally (they can be pedagogically useful),
        # but avoid making them dominant.
        if lhs == rhs and random.random() < 0.8:
            alternatives = [c for c in state['numeric_vars'] if c != lhs]
            if alternatives:
                rhs = random.choice(alternatives)
        return f"{lhs} {op} {rhs}"

    # -------------------------------------------------------------------------
    # 3. Grammar Definition
    # -------------------------------------------------------------------------
    R('CTX', '') 
    R('RESET(CTX)', reset_state)

    # Terminals
    R('DIGIT', lambda: str(random.randint(0, max_number)))
    R('SMALL_INDEX', lambda: str(random.randint(0, 1)))
    R('STR_LIT', lambda: random.choice(['"hi"', '"cat"', '"go"', '"sun"']))
    R('LIST_LIT(DIGIT, DIGIT, DIGIT)', '[0, 1, 2]')
    R('VAR(CTX)', lambda x: random.choice(chars))
    for op in ['+', '-', '*']: R('ARITH_OP', op)
    for op in ['<', '>', '<=', '>=', '!=', '==']: R('REL_OP', op)
    R('LOG_PREFIX', 'not ')
    for op in ['and', 'or']: R('LOG_INFIX', op)

    # Expressions
    R('EXPR_ID(CTX)', get_assigned_var)
    R('ATOM(CTX)', get_atom)
    R('TERM(EXPR_ID)', '0'); R('TERM(DIGIT)', '0')
    R('TERM(ATOM)', '0')
    R('EXPRESSION(TERM, ARITH_OP, TERM)', '0 1 2')
    R('EXPRESSION(TERM)', '0', weight=0.35)
    R('ENCLOSED(EXPRESSION)', '(0)')

    R('DISP_ID(CTX)', get_last_var)
    R('DISP_EXPR(EXPR_ID, ARITH_OP, EXPR_ID)', '0 1 2')
    R('DISP_EXPR(EXPR_ID, ARITH_OP, DIGIT)', '0 1 2')
    R('LEN_EXPR(STR_LIT)', 'len(0)')
    R('LEN_EXPR(LIST_LIT)', 'len(0)')
    R('INDEX_EXPR(LIST_LIT, SMALL_INDEX)', '0[1]')
    R('STR_INDEX_EXPR(STR_LIT, SMALL_INDEX)', '0[1]')

    # Initializations
    R('INIT(CTX)', render_init)
    # Unrolled max_init logic
    for n in [2, 3, 4, 5]:
        for k in range(1, n + 2):
            R(f"IDENT_INIT_{n}(" + ",".join(["INIT"]*k) + ")", concat)

    # Assignments
    R('SIMPLE_ARITH(ENCLOSED)', '0')
    R('SIMPLE_ARITH(SIMPLE_ARITH, ARITH_OP, ENCLOSED)', concat, weight=0.5)
    
    R('SIMPLE_ASSIGN(VAR, EXPRESSION)', render_assign)
    R('SIMPLE_ASSIGNS', '')
    R('SIMPLE_ASSIGNS(SIMPLE_ASSIGN)', '0')

    R('ADV_ASSIGN_TYPE(VAR, SIMPLE_ARITH)', render_assign)
    R('ADV_ASSIGN_TYPE(VAR, EXPRESSION)', render_assign)
    R('ADV_ASSIGN_TYPE(VAR, STR_LIT)', lambda v, e: render_assign(v, e, kind='str'))
    R('ADV_ASSIGN_TYPE(VAR, LIST_LIT)', lambda v, e: render_assign(v, e, kind='list'))
    R('ADV_ASSIGN_TYPE(VAR, LEN_EXPR)', render_assign)
    R('ADV_ASSIGN_TYPE(VAR, INDEX_EXPR)', render_assign)
    R('ADV_ASSIGNS', '')
    R('ADV_ASSIGNS(ADV_ASSIGN_TYPE)', '0')

    # Conditions
    R('COND_EXPR(EXPR_ID, REL_OP, EXPR_ID)', render_cond_expr)
    R('COND_EXPR(EXPR_ID, REL_OP, DIGIT)', '0 1 2')
    R('COND(COND_EXPR)', '0')
    R('COND(LOG_PREFIX, COND_EXPR)', '01')
    R('ENCLOSED_COND(COND)', '(0)')
    R('CHAIN(ENCLOSED_COND)', '0')
    R('CHAIN(LOG_PREFIX, ENCLOSED_COND)', '01')
    R('CHAIN(CHAIN, LOG_INFIX, ENCLOSED_COND)', '0 1 2', weight=0.3)

    R('IF_BLK(COND)', 'if 0:\n')
    R('ELIF_BLK(COND)', 'elif 0:\n')
    R('ELSE_BLK', 'else:\n')
    R('ADV_IF(CHAIN)', 'if 0:\n')
    R('ADV_ELIF(CHAIN)', 'elif 0:\n')

    # Display
    R('DISPLAY(DISP_ID)', 'print(0)')
    R('ADV_DISP(DISPLAY)', '0')
    R('ADV_DISP(DISP_EXPR)', 'print(0)')
    R('ADV_DISP(STR_LIT)', 'print(0)')
    R('ADV_DISP(LIST_LIT)', 'print(0)')
    R('ADV_DISP(LEN_EXPR)', 'print(0)')
    R('ADV_DISP(INDEX_EXPR)', 'print(0)')
    R('ADV_DISP(STR_INDEX_EXPR)', 'print(0)')
    R('BODY_STMT(DISPLAY)', '0')
    R('BODY_STMT(SIMPLE_ASSIGN)', '0')
    R('BODY_STMT(ADV_ASSIGN_TYPE)', '0')

    # Loops
    R('FOR_INIT(CTX)', lambda x: render_loop_math(x, 'init'))
    R('FOR_FINAL(CTX)', lambda x: render_loop_math(x, 'final_less'))
    R('STEP(CTX)', lambda x: state['loops'].get('step', '1'))
    
    R('FOR_HEAD(VAR, FOR_INIT, FOR_FINAL, STEP)', 'for 0 in range(1, 2, 3):')
    R('FOR_HEAD(VAR, FOR_INIT, FOR_FINAL)', 'for 0 in range(1, 2):')
    R('FOR_LOOP(FOR_HEAD, BODY_STMT)', '0\n\t1')
    R('ADV_FOR(FOR_HEAD, BODY_STMT)', '0\n\t1')
    R('ADV_FOR(FOR_LOOP)', '0')

    R('REL_LESS', '<'); R('REL_LESS', '<=')
    R('REL_GREATER', '>'); R('REL_GREATER', '>=')
    R('WHILE_VAR(CTX)', render_while_var)
    R('WH_FINAL_L(CTX)', lambda x: render_loop_math(x, 'final_less'))
    R('WH_FINAL_G(CTX)', lambda x: render_loop_math(x, 'final_greater'))
    R('WH_UPD_L(CTX)', lambda x: render_while_update(x, '+'))
    R('WH_UPD_G(CTX)', lambda x: render_while_update(x, '-'))

    R('WH_L(WHILE_VAR, REL_LESS, WH_FINAL_L, DISPLAY, WH_UPD_L)', 'while 0 1 2:\n\t3\n\t4')
    R('WH_G(WHILE_VAR, REL_GREATER, WH_FINAL_G, DISPLAY, WH_UPD_G)', 'while 0 1 2:\n\t3\n\t4')

    # Levels
    R('L1_1(IDENT_INIT_2, SIMPLE_ASSIGNS, ADV_DISP)', concat)
    R('L1_2(IDENT_INIT_3, ADV_ASSIGNS, ADV_DISP)', concat)

    R('L2_1(IDENT_INIT_2, IF_BLK, DISPLAY)', 
      lambda i, f, d: f"{i.render('py')}{f.render('py')}\t{d.render('py')}")
    R('L2_1(IDENT_INIT_2, IF_BLK, DISPLAY, ELIF_BLK, DISPLAY, ELSE_BLK, DISPLAY)', 
      lambda i, if_, d1, el_, d2, els, d3: f"{i.render('py')}{if_.render('py')}\t{d1.render('py')}\n{el_.render('py')}\t{d2.render('py')}\n{els.render('py')}\t{d3.render('py')}")
    R('L2_1(IDENT_INIT_2, IF_BLK, DISPLAY, ELSE_BLK, DISPLAY)', 
      lambda i, if_, d1, els, d2: f"{i.render('py')}{if_.render('py')}\t{d1.render('py')}\n{els.render('py')}\t{d2.render('py')}")

    R('L2_2(IDENT_INIT_5, ADV_ASSIGNS, ADV_IF, ADV_DISP)', 
      lambda i, a, f, d: f"{i.render('py')}{a.render('py')}{f.render('py')}\t{d.render('py')}")
    R('L2_2(IDENT_INIT_5, ADV_ASSIGNS, ADV_IF, ADV_DISP, ADV_ELIF, ADV_DISP, ELSE_BLK, ADV_DISP)', 
      lambda *x: f"{x[0].render('py')}{x[1].render('py')}{x[2].render('py')}\t{x[3].render('py')}\n{x[4].render('py')}\t{x[5].render('py')}\n{x[6].render('py')}\t{x[7].render('py')}")
    R('L2_2(IDENT_INIT_5, ADV_ASSIGNS, ADV_IF, ADV_DISP, ELSE_BLK, ADV_DISP)', 
      lambda *x: f"{x[0].render('py')}{x[1].render('py')}{x[2].render('py')}\t{x[3].render('py')}\n{x[4].render('py')}\t{x[5].render('py')}")

    R('L3_1(IDENT_INIT_2, FOR_LOOP)', concat)
    R('L3_2(IDENT_INIT_4, ADV_ASSIGNS, ADV_FOR)', concat)
    R('L4_1(IDENT_INIT_2, WH_L)', concat)
    R('L4_1(IDENT_INIT_2, WH_G)', concat)

    # Root - CRITICAL: Call r.render() to trigger RESET side-effects
    valid_levels = ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1"]
    selection = [level] if level in valid_levels else valid_levels

    for l in selection:
        rule_name = f"L{l.replace('.', '_')}"
        R(f'ALL(RESET, {rule_name})', lambda r, lvl: r.render('py') + lvl.render('py'))

    R('start(ALL)', '0')

    return R
