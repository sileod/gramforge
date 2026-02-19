import copy
from collections import deque
from functools import lru_cache
import random
import re
from easydict import EasyDict as edict
import numpy as np
from gramforge.grammar import Substitution

class FastProduction:
    """
    A unified, high-performance production node.
    """
    __slots__ = ('rule', 'type', 'parent', 'children', 'state', 'cache', 
                 '_depth', '_height')

    def __init__(self, rule=None, type=None, parent=None, children=None):
        self.rule = rule or ''
        self.type = type or (rule.name if rule else '')
        self.parent = parent
        
        self.cache = {}
 
        if children is not None:
            self.children = children
        elif self.rule:

            self.children = [
                FastProduction(type=arg, parent=self) 
                for arg in self.rule.args
            ]
        else:
            self.children = []

        self._depth = -1 
        self._height = -1
 

    def clone(self, memo=None):
        if memo is None: memo = {}
        if id(self) in memo: return memo[id(self)]
        new_obj = FastProduction(rule=self.rule, type=self.type)
        memo[id(self)] = new_obj
        new_obj.children = [child.clone(memo) for child in self.children]
        for child in new_obj.children:
            child.parent = new_obj
        return new_obj

    def check(self, mode=None):
        return True # Placeholder for your real constraint logic

    def update_height(self):
        if self._height != -1: return self._height
        self._height = 0 if not self.children else 1 + max((c.update_height() for c in self.children), default=0)
        return self._height
    @property
    def height(self):
        if self._height == -1: self.update_height()
        return self._height
    @property
    def depth(self):
        if self._depth != -1: return self._depth
        if self.parent is None: self._depth = 0
        else: self._depth = self.parent.depth + 1
        return self._depth
    @property
    def leaves(self):
        if not self.children: return [self]
        leaves_list, q = [], deque([self])
        while q:
            node = q.popleft()
            if not node.children: leaves_list.append(node)
            else: q.extendleft(reversed(node.children))
        return leaves_list
    @property
    def ancestors(self):
        anc, node = [], self.parent
        while node: anc.append(node); node = node.parent
        return tuple(anc)
    @property
    def siblings(self):
        return tuple(c for c in self.parent.children if c is not self) if self.parent else tuple()
    @property
    def descendants(self):
        if not self.children: return tuple()
        desc, q = [], deque(self.children)
        while q:
            node = q.popleft(); desc.append(node); q.extend(node.children)
        return tuple(desc)
    def render(self, lang=None):
        if lang in self.cache: return self.cache[lang]
        if not self.rule: return "#" + self.type
        template = self.rule.templates.get(lang)
        if template is None: return "#" + self.type
        if callable(template): args, template_func = self.children, template
        else:
            args = [c.render(lang) for c in self.children]
            if '?←' in template: template_func = Substitution(template, lang)
            else: template_func = template.format
        out = template_func(*args);
        if "#" not in out: self.cache[lang] = out
        return out
    __getitem__ = lambda self, key: self.children[key]
    __matmul__ = render
    __repr__ = lambda self: f"PROD:{self.type}" + (f"({', '.join(self.rule.args)})" if self.rule else "")

    def dict(self, use_cls=False):
        res = edict({l: self@l for l in self.rule.langs})
        if use_cls: res.cls = self
        return res



@lru_cache(maxsize=None)
def _precompute_height_bounds(Rule):
    """Computes min/max possible height for any non-terminal. Used by both modes."""
    non_terminals = list({r.name for r in Rule._instances})
    min_h, max_h = {nt: 999 for nt in non_terminals}, {nt: 0 for nt in non_terminals}
    for r in Rule._instances:
        if not r.args: min_h[r.name], max_h[r.name] = 0, 0
    for _ in range(len(non_terminals) + 1):
        for r in Rule._instances:
            if not r.args: continue
            child_mins = [min_h.get(arg, 999) for arg in r.args]
            child_maxs = [max_h.get(arg, 0) for arg in r.args]
            if all(h != 999 for h in child_mins):
                min_h[r.name] = min(min_h[r.name], 1 + max(child_mins))
            if all(h != 0 or max_h.get(arg, -1) !=-1 for arg, h in zip(r.args, child_maxs)): # handle cycles
                 max_h[r.name] = max(max_h[r.name], 1 + max(child_maxs))
    return min_h, max_h


def generate_recursive(start, depth=12, min_depth=None, n_iter=1000, bushiness_factor=0.5, **kwargs):
    """
    A fast, state-less generator that offers fine-grained control over tree topology.
    This version uses a true recursive helper function to avoid tree connection bugs.
    MODIFIED to perform weighted sampling of rules based on their .weight attribute.
    """
    Rule = type(start)
    min_heights, _ = _precompute_height_bounds(Rule)
    non_terminal_types = set(min_heights.keys())
    max_depth, min_depth = depth, min_depth or 0
    bushiness_factor = max(0.0, min(1.0, bushiness_factor))

    class _Node:
        __slots__ = ['rule', 'type', 'depth', 'children']
        def __init__(self, type, depth): self.type, self.depth, self.children = type, depth, []
        def __repr__(self): return f"_{self.type}(d={self.depth})"
        
    def _select_rule(rules):
        """Performs a weighted sample from a list of rules."""
        # This function assumes 'rules' is a non-empty list.
        # It defaults to a weight of 1.0 if the attribute is missing.
        weights = [getattr(r, 'weight', 1.0) for r in rules]
        
        # random.choices fails if all weights are 0. Fallback to uniform.
        if sum(weights) <= 0:
            return random.choice(rules)
            
        return random.choices(rules, weights=weights, k=1)[0]

    # --- The Recursive Tree-Building Helper Function ---
    def _build_tree(node, min_height_target):
        # Base case 1: Pruning - Impossible to meet target within max_depth
        if min_height_target > max_depth - node.depth:
            return False # Failure

        # Base case 2: Target met, fill randomly respecting only max_depth
        if min_height_target <= 0:
            potential_rules = [r for r in Rule.get_rules(node.type)
                               if node.depth + min_heights.get(r.name, 0) <= max_depth]
            if not potential_rules:
                return False # Failure
            
            node.rule = _select_rule(potential_rules)

            for arg in node.rule.args:
                child_node = _Node(type=arg, depth=node.depth + 1)
                node.children.append(child_node)
                if not _build_tree(child_node, 0): # Recurse with target 0
                    return False # Propagate failure
            return True # Success

        # Recursive Step: We have a depth target to meet-
        valid_rules = [r for r in Rule.get_rules(node.type)
                       if r.args and any(arg in non_terminal_types for arg in r.args)]
        if not valid_rules:
            return False # Failure
        
        node.rule = _select_rule(valid_rules)
        
        child_nodes = [_Node(type=arg, depth=node.depth + 1) for arg in node.rule.args]
        node.children = child_nodes
        
        potential_spine_indices = [i for i, child in enumerate(child_nodes) if child.type in non_terminal_types]
        if not potential_spine_indices: # All children are terminals, cannot continue spine
             for child_node in child_nodes:
                if not _build_tree(child_node, 0): return False
             return True

        spine_child_idx = random.choice(potential_spine_indices)
        
        new_spine_target = min_height_target - 1
        new_side_branch_target = int(np.floor(new_spine_target * bushiness_factor))
        
        success = True
        for i, child_node in enumerate(child_nodes):
            target = new_spine_target if i == spine_child_idx else new_side_branch_target
            if not _build_tree(child_node, target):
                success = False
                break
        return success

    # --- Main Generation Loop ---
    for _ in range(n_iter):
        root = _Node(type=start.name, depth=0)
        
        # Start the recursive generation process
        if _build_tree(root, min_depth):
            # If successful, convert the _Node tree to a FastProduction tree
            
            # Helper to collect all nodes from the valid _Node tree
            def collect_nodes(node):
                all_nodes = [node]
                for child in node.children:
                    all_nodes.extend(collect_nodes(child))
                return all_nodes

            # Now do the final conversion
            node_map = {}
            for node in reversed(collect_nodes(root)): # Post-order traversal
                prod = FastProduction(rule=node.rule, type=node.type, parent=None)
                children = [node_map[id(c)] for c in node.children]
                prod.children = children
                for child in children:
                    child.parent = prod
                node_map[id(node)] = prod
            
            return [node_map[id(root)]]

    return []