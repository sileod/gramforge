# generate_choices.py
"""
Generate partial derivations and return valid next tokens.
Uses bounded search to find exact set of valid next tokens.
"""
import re
import random
from typing import Optional, Union, Tuple, Set
from functools import lru_cache
from easydict import EasyDict as edict
from .generate_sequential import FastProduction, _precompute_height_bounds

STOP_TOKEN = '[STOP]'
_PLACEHOLDER_RE = re.compile(r'^[\(\[\{<"\'`.,;:]*#')


def generate_with_choices(
    start,
    target_n_choices: Optional[Union[int, Tuple[int, int], Set[int]]] = None,
    max_depth: int = 12,
    min_depth: Optional[int] = None,
    min_prefix_len: int = 1,
    max_prefix_len: Optional[int] = None,
    bushiness: float = 1.0,
    stop_prob: float = 0.3,
    max_steps: int = 5000,
    lang: Optional[str] = None,
    include_stop: bool = True,
    skip_check: bool = False,
    **kwargs
) -> Optional[edict]:
    """
    Generate a partial parse tree and return valid next token choices.
    
    Uses bounded search to find the exact set of terminal tokens that could
    legally appear next in the derivation.
    
    Args:
        start: Rule instance (grammar start symbol)
        target_n_choices: Filter by number of choices (int, (min,max) tuple, or set)
        max_depth: Maximum tree height
        min_depth: Minimum tree height
        min_prefix_len: Minimum tokens before returning
        max_prefix_len: Maximum tokens
        bushiness: Controls tree shape (0=vine-like, 1=bushy)
        stop_prob: Probability of returning at valid state
        max_steps: Maximum expansion steps
        lang: Template language key
        include_stop: Include [STOP] in choices when complete
        skip_check: Skip constraint validation
    
    Returns:
        EasyDict with: prefix, choices, n_choices, is_complete, tree, depth, current_type
        Or None if generation fails.
    """
    min_depth = 0 if min_depth is None else min_depth
    if min_depth > max_depth:
        return None

    # Resolve Rule class and create start production
    Rule = kwargs.pop("Rule", None)
    
    # Handle if start is a Rule class (type) rather than instance
    if type(start) == type and hasattr(start, 'start'):
        Rule = start
        start = start.start()
    
    if isinstance(start, FastProduction):
        start_prod = start
        if Rule is None and start_prod.rule:
            Rule = type(start_prod.rule)
    elif isinstance(start, str):
        if Rule is None:
            raise ValueError("If start is a string, must pass Rule=")
        start_prod = FastProduction(type=start)
    else:
        Rule = type(start)
        start_prod = FastProduction(start)

    if Rule is None:
        return None

    min_heights, max_heights = _precompute_height_bounds(Rule)

    # Default language
    if lang is None and hasattr(start, "langs") and getattr(start, "langs", None):
        lang = start.langs[0]

    start_prod.local_state["target_min_depth"] = min_depth

    # --- Helper functions ---
    
    def is_placeholder(tok: str) -> bool:
        """Check if token is an unexpanded placeholder like #Type"""
        return bool(_PLACEHOLDER_RE.match(tok))

    def tokenize(s: str) -> list:
        """Split string into whitespace-separated tokens"""
        return re.findall(r"\S+", s) if s else []

    def get_prefix(prod: FastProduction) -> tuple:
        """Extract prefix tokens and check if tree is complete"""
        rendered = prod.render(lang)
        tokens = tokenize(rendered)
        prefix = []
        complete = True
        for tok in tokens:
            if is_placeholder(tok):
                complete = False
                break
            prefix.append(tok)
        if prod._find_first_unexpanded_leaf() is not None:
            complete = False
        return prefix, complete

    def matches_target(n: int) -> bool:
        """Check if n matches target_n_choices specification"""
        if target_n_choices is None:
            return True
        if isinstance(target_n_choices, int):
            return n == target_n_choices
        if isinstance(target_n_choices, tuple) and len(target_n_choices) == 2:
            return target_n_choices[0] <= n <= target_n_choices[1]
        if isinstance(target_n_choices, set):
            return n in target_n_choices
        return True

    def get_path(node: FastProduction) -> list:
        """Get path from root to node as list of child indices"""
        path = []
        curr = node
        while curr.parent:
            path.append(curr.parent.children.index(curr))
            curr = curr.parent
        return list(reversed(path))

    def locate_by_path(root: FastProduction, path: list) -> FastProduction:
        """Find node at given path from root"""
        node = root
        for idx in path:
            node = node.children[idx]
        return node

    def get_valid_rules(lv: FastProduction) -> tuple:
        """Get valid rules for leaf, split into non-terminals and terminals"""
        target = lv.state.get("target_min_depth", 0)
        rules = Rule.get_rules(lv.type, shuffle=True)
        non_terms, terms = [], []
        
        for r in rules:
            if r.args:
                min_h = 1 + max((min_heights.get(a, max_depth + 1) for a in r.args), default=0)
                max_h = 1 + max((max_heights.get(a, 0) for a in r.args), default=0)
            else:
                min_h, max_h = 0, 0

            # Depth pruning
            if lv.depth + min_h > max_depth:
                continue
            if lv.depth + max_h < target:
                continue

            if r.args:
                non_terms.append(r)
            elif lv.depth >= target:
                terms.append(r)

        return non_terms, terms, target

    def apply_rule(root: FastProduction, path: list, rule, target: int, 
                   crit_idx: Optional[int] = None, deterministic: bool = False) -> bool:
        """Apply rule at leaf specified by path. Returns success."""
        lv = locate_by_path(root, path)
        lv.rule = rule
        if rule.state:
            lv.local_state.update(rule.state)

        if rule.args:
            # Find critical child that must maintain depth constraint
            crits = [
                i for i, a in enumerate(rule.args)
                if lv.depth + 1 + max_heights.get(a, 0) >= target
            ]
            if not crits:
                return False
            if crit_idx is None:
                crit_idx = random.choice(crits)
            elif crit_idx not in crits:
                return False

            children = []
            for i, arg_type in enumerate(rule.args):
                child = FastProduction(type=arg_type, parent=lv)
                if i == crit_idx:
                    child.local_state["target_min_depth"] = target
                else:
                    if deterministic:
                        child.local_state["target_min_depth"] = child.depth
                    else:
                        remaining = max(0, target - child.depth)
                        addon = int(bushiness * remaining)
                        child.local_state["target_min_depth"] = child.depth + random.randint(0, addon)
                children.append(child)
            lv.children = children
        else:
            lv.children = []

        # Constraint checks
        if not skip_check:
            if not lv.check("state"):
                return False
            if lv.parent and all(s.rule for s in lv.siblings) and not lv.parent.check("args"):
                return False

        return True

    def compute_next_choices(prod: FastProduction) -> tuple:
        """
        Bounded search to find all valid next tokens.
        Returns (choices, current_type, current_depth).
        """
        base_prefix, _ = get_prefix(prod)
        base_len = len(base_prefix)
        choices = set()
        
        budget = int(kwargs.get("choice_budget", 1200))
        stack = [prod.clone()]

        while stack and budget > 0:
            budget -= 1
            state = stack.pop()
            current_prefix, complete = get_prefix(state)

            # Found a new token
            if len(current_prefix) > base_len:
                choices.add(current_prefix[base_len])
                continue

            # Tree is complete
            if complete:
                if include_stop:
                    choices.add(STOP_TOKEN)
                continue

            # Find unexpanded leaf and expand all valid ways
            lv = state._find_first_unexpanded_leaf()
            if lv is None:
                if include_stop:
                    choices.add(STOP_TOKEN)
                continue

            non_terms, terms, target = get_valid_rules(lv)
            path = get_path(lv)

            for r in non_terms + terms:
                if r.args:
                    # Try all critical child indices
                    crits = [
                        i for i, a in enumerate(r.args)
                        if lv.depth + 1 + max_heights.get(a, 0) >= target
                    ]
                    for ci in crits:
                        new_state = state.clone()
                        if apply_rule(new_state, path, r, target, crit_idx=ci, deterministic=True):
                            stack.append(new_state)
                else:
                    new_state = state.clone()
                    if apply_rule(new_state, path, r, target, deterministic=True):
                        stack.append(new_state)

        # Sort with STOP at end
        result = sorted(x for x in choices if x != STOP_TOKEN)
        if include_stop and STOP_TOKEN in choices:
            result.append(STOP_TOKEN)

        lv0 = prod._find_first_unexpanded_leaf()
        current_type = lv0.type if lv0 else None
        current_depth = lv0.depth if lv0 else prod.height

        return result, current_type, current_depth

    # --- Main generation loop ---
    
    stack = [start_prod]
    step = 0

    while stack:
        step += 1
        if step > max_steps:
            return None

        prod = stack.pop()

        # Prune by max prefix length
        if max_prefix_len is not None:
            prefix, _ = get_prefix(prod)
            if len(prefix) > max_prefix_len:
                continue

        # Check if we should return at this state
        prefix, is_complete = get_prefix(prod)
        prefix_len = len(prefix)

        if prefix_len >= min_prefix_len and (max_prefix_len is None or prefix_len <= max_prefix_len):
            choices, current_type, depth = compute_next_choices(prod)
            n_choices = len(choices)

            if n_choices > 0 and matches_target(n_choices):
                # Force stop at max_prefix_len, otherwise use probability
                force_stop = max_prefix_len is not None and prefix_len >= max_prefix_len
                if force_stop or random.random() < stop_prob:
                    return edict(
                        prefix=prefix,
                        choices=choices,
                        n_choices=n_choices,
                        is_complete=is_complete or current_type is None,
                        tree=prod,
                        depth=depth,
                        current_type=current_type
                    )

        # Expand tree
        lv = prod._find_first_unexpanded_leaf()
        if lv is None:
            continue

        non_terms, terms, target = get_valid_rules(lv)

        # Bushiness controls preference for terminals
        can_terminate = len(terms) > 0
        has_met_goal = lv.depth >= target

        if has_met_goal and can_terminate and random.random() > bushiness:
            rules = terms
        else:
            rules = non_terms + terms

        if not rules:
            continue

        # Weighted selection
        weights = [r.weight for r in rules]
        if sum(weights) > 0:
            rule = random.choices(rules, weights=weights, k=1)[0]
        else:
            rule = random.choice(rules)

        path = get_path(lv)
        if apply_rule(prod, path, rule, target):
            stack.append(prod)

    return None


def generate_choices_batch(start, n=10, **kwargs):
    """Generate multiple choice examples. Simple wrapper around generate_with_choices."""
    results = []
    for _ in range(n * 10):  # Try more times than needed
        result = generate_with_choices(start, **kwargs)
        if result:
            results.append(result)
            if len(results) >= n:
                break
    return results
