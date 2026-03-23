import random
from .generate_sequential import FastProduction, _precompute_height_bounds


def _collect_unexpanded_paths(root):
    paths = []
    stack = [(root, ())]
    while stack:
        node, path = stack.pop()
        if not node.rule:
            paths.append(path)
            continue
        for idx in range(len(node.children) - 1, -1, -1):
            stack.append((node.children[idx], path + (idx,)))
    return tuple(paths)


def _locate_by_path(root, path):
    node = root
    for idx in path:
        node = node.children[idx]
    return node


def _get_inherited(node, key, default=None):
    while node is not None:
        if key in node.local_state:
            return node.local_state[key]
        node = node.parent
    return default


def generate_sequential_opt(start, k=1, max_depth=12, min_depth=None, bushiness=1.0,
                            skip_check=False, max_steps=5000, save_prob=0.0125,
                            debug=False, **kwargs):
    min_depth = min_depth or 0
    bushiness = max(0.0, min(1.0, bushiness))
    if min_depth > max_depth:
        return []

    Rule = type(start)
    min_heights, max_heights = _precompute_height_bounds(Rule)
    rules_by_name = {}
    for rule in Rule._instances:
        rules_by_name.setdefault(rule.name, []).append(rule)

    def save(item, stack, st):
        prod, frontier = item
        ckpt = prod.clone()
        ckpt.step, ckpt.save = st, 1
        insert_at = ([0] + [i for i, x in enumerate(stack) if hasattr(x[0], 'save') and x[0].save])[-1]
        stack.insert(insert_at, (ckpt, frontier))

    def s_choices(seq, weights, n):
        if not seq:
            return []
        if weights and sum(weights) > 0:
            return random.choices(seq, weights=weights, k=n)
        return random.sample(seq, min(n, len(seq)))

    start_prod = FastProduction(start)
    start_prod.local_state['target_min_depth'] = min_depth
    start_prod.step, start_prod.save = 0, 0

    stack, step = [(start_prod, _collect_unexpanded_paths(start_prod))], 0
    while stack:
        step += 1
        if step > max_steps:
            return []
        prod, frontier = stack.pop()

        if not frontier:
            prod.update_height()
            if min_depth <= prod.height <= max_depth and \
               (skip_check or all(x.check() for x in [prod] + list(prod.descendants))):
                return [prod]
            continue

        path = frontier[0]
        lv = _locate_by_path(prod, path)
        current_target_min_depth = _get_inherited(lv, 'target_min_depth', 0)
        if debug:
            print(f"[{step}] Expanding '{lv.type}' @ d={lv.depth} (goal_min={current_target_min_depth}) | Stack: {len(stack)}")

        potential_rules = list(rules_by_name.get(lv.type, ()))
        random.shuffle(potential_rules)
        valid_non_terminals, valid_terminals = [], []

        for rule in potential_rules:
            min_h_rule = 0 if not rule.args else 1 + max((min_heights.get(a, max_depth + 1) for a in rule.args), default=0)
            max_h_rule = 0 if not rule.args else 1 + max((max_heights.get(a, 0) for a in rule.args), default=0)

            if lv.depth + min_h_rule > max_depth:
                continue
            if lv.depth + max_h_rule < current_target_min_depth:
                continue

            if rule.args:
                valid_non_terminals.append(rule)
            elif lv.depth >= current_target_min_depth:
                valid_terminals.append(rule)

        can_terminate = bool(valid_terminals)
        has_met_goal = lv.depth >= current_target_min_depth
        if has_met_goal and can_terminate and random.random() > bushiness:
            rules_to_consider = valid_terminals
        else:
            rules_to_consider = valid_non_terminals + valid_terminals

        if debug:
            print(f"  Candidates ({len(rules_to_consider)}): {[r.name for r in rules_to_consider]}")
        if not rules_to_consider:
            continue

        rules_to_try = s_choices(rules_to_consider, [r.weight for r in rules_to_consider], k)
        if len(rules_to_try) > 1 and random.random() < save_prob:
            save((prod, frontier), stack, step)

        tail_frontier = frontier[1:]
        for i, rule in enumerate(rules_to_try):
            branch = prod if i == len(rules_to_try) - 1 else prod.clone()
            target_lv = _locate_by_path(branch, path)
            target_lv.rule = rule
            if rule.state:
                target_lv.local_state.update(rule.state)

            if rule.args:
                critical_candidates = [
                    idx for idx, arg_type in enumerate(rule.args)
                    if target_lv.depth + 1 + max_heights.get(arg_type, 0) >= current_target_min_depth
                ]
                if not critical_candidates:
                    continue

                critical_idx = random.choice(critical_candidates)
                new_children = []
                for idx, arg_type in enumerate(rule.args):
                    child = FastProduction(type=arg_type, parent=target_lv)
                    if idx == critical_idx:
                        child.local_state['target_min_depth'] = current_target_min_depth
                    else:
                        remaining_depth = max(0, current_target_min_depth - child.depth)
                        bushy_addon = int(bushiness * remaining_depth)
                        child.local_state['target_min_depth'] = child.depth + random.randint(0, bushy_addon)
                    new_children.append(child)
                target_lv.children = new_children
                next_frontier = tuple(path + (idx,) for idx in range(len(rule.args))) + tail_frontier
            else:
                target_lv.children = []
                next_frontier = tail_frontier

            if not skip_check and (
                not target_lv.check('state') or
                (target_lv.parent and all(s.rule for s in target_lv.siblings) and not target_lv.parent.check('args'))
            ):
                continue

            stack.append((branch, next_frontier))
            break

    return []
