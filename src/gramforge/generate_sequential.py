# generate.py
import random
from collections import deque, ChainMap
from functools import lru_cache
import anytree
from easydict import EasyDict as edict
# Assuming .grammar and other imports are correctly set up in the project structure
from .grammar import Substitution

import re

# ... (FastProduction class and _precompute_height_bounds function remain unchanged) ...
class FastProduction:
    """A unified, high-performance production node for all generation modes."""
    __slots__ = ('rule', 'type', 'parent', 'children', 'local_state', 'cache', '_depth', 'step', 'save', '_height')

    def __init__(self, rule=None, type=None, parent=None, children=None, state=None):
        self.rule, self.type, self.parent, self.cache = rule or '', type or (rule.name if rule else ''), parent, {}
        self.children = children if children is not None else []
        self._depth = (parent.depth + 1 if parent else 0)
        self._height = -1
        self.local_state = state or {}

        if self.rule and not children:
            if self.rule.state:
                self.local_state = {**self.rule.state, **self.local_state}
            self.children = [FastProduction(type=arg, parent=self) for arg in self.rule.args]
            
    @property
    def state(self):
        if self.parent:
            return ChainMap(self.local_state, self.parent.state)
        return self.local_state

    def clone(self):
        new_root = FastProduction(rule=self.rule, state=self.local_state.copy())
        if hasattr(self, 'step'): new_root.step = self.step
        if hasattr(self, 'save'): new_root.save = self.save
        q_orig, q_clone = deque([self]), deque([new_root])
        while q_orig:
            orig_node, clone_node = q_orig.popleft(), q_clone.popleft()
            clone_node.children = [FastProduction(rule=c.rule, type=c.type, parent=clone_node, state=c.local_state.copy()) for c in orig_node.children]
            q_orig.extend(orig_node.children); q_clone.extend(clone_node.children)
        return new_root
    
    def _find_first_unexpanded_leaf(self):
        q = deque([self])
        while q:
            node = q.popleft()
            if not node.rule:
                return node
            q.extendleft(reversed(node.children))
        return None

    def update_height(self):
        if self._height != -1: return self._height
        self._height = 0 if not self.children else 1 + max((c.update_height() for c in self.children), default=-1)
        return self._height

    @property
    def height(self):
        if self._height == -1: self.update_height()
        return self._height
        
    @property
    def depth(self): return self._depth
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

    def check(self, mode='args'):
        if not self.rule: return True
        if mode == 'args':
            if not all(x.rule for x in self.children): return True
            return all(c(self.children) for c in self.rule.constraint)
        if mode == 'state':
            return all(c(x) for x in [self] + list(self.ancestors) for c in x.rule.state_constraint)
        return True

    def render(self, lang=None):
        if lang in self.cache: return self.cache[lang]
        if not self.rule: return "#" + self.type
        template = self.rule.templates.get(lang)
        if template is None: return "#" + self.type
        
        if callable(template):
            args, template_func = self.children, template
        else: # String template
            if '?←' in template:
                args = self.children
                template_func = Substitution(template, lang)
            else:
                args = [c.render(lang) for c in self.children]
                template_func = template.format

        out = template_func(*args)
        if "#" not in out: self.cache[lang] = out
        return out

    def __getitem__(self, key): return self.children[key]
    __matmul__ = render

    def __repr__(self):
        if self.rule: return f"PROD:{self.type}({', '.join(self.rule.args)})"
        else: return f"PROD:{self.type}"
    
    def dict(self, use_cls=False):
        res = edict({l: self@l for l in self.rule.langs})
        if use_cls: res.cls = self
        return res

    def to_anytree(self):
        memo = {}
        def _convert_node(node):
            if id(node) in memo: return memo[id(node)]
            new_node = anytree.Node(node.type)
            new_node.rule = node.rule
            new_node.state = dict(node.state) 
            for child in node.children:
                _convert_node(child).parent = new_node
            memo[id(node)] = new_node
            return new_node
        return _convert_node(self)

@lru_cache(maxsize=None)
def _precompute_height_bounds(Rule):
    non_terminals = list({r.name for r in Rule._instances})
    min_h, max_h = {nt: 999 for nt in non_terminals}, {nt: 0 for nt in non_terminals}
    for r in Rule._instances:
        if not r.args: min_h[r.name], max_h[r.name] = 0, 0
    for _ in range(len(non_terminals) + 1):
        for r in Rule._instances:
            if not r.args: continue
            child_mins = [min_h.get(arg, 999) for arg in r.args]
            child_maxs = [max_h.get(arg, 0) for arg in r.args]
            if all(h != 999 for h in child_mins): min_h[r.name] = min(min_h[r.name], 1 + max(child_mins))
            if all(h != 0 or max_h.get(arg, -1) != -1 for arg, h in zip(r.args, child_maxs)):
                max_h[r.name] = max(max_h[r.name], 1 + max(child_maxs))
    min_h = {k: v for k, v in min_h.items() if v != 999}
    return min_h, max_h


def generate_sequential(start, k=1, max_depth=12, min_depth=None, bushiness=1.0,
                        skip_check=False, max_steps=5000, save_prob=0.0125, debug=False, **kwargs):
    min_depth = min_depth or 0
    if min_depth > max_depth: return []

    Rule = type(start)
    min_heights, max_heights = _precompute_height_bounds(Rule)

    def save(p, s, st):
        ckpt = p.clone(); ckpt.step, ckpt.save = st, 1
        s.insert(([0] + [i for i, x in enumerate(s) if hasattr(x, 'save') and x.save])[-1], ckpt)

    def s_choices(seq, w, k):
        if not seq: return []
        if w and sum(w) > 0: return random.choices(seq, weights=w, k=k)
        return random.sample(seq, min(k, len(seq)))

    start_prod = FastProduction(start)
    start_prod.local_state['target_min_depth'] = min_depth
    start_prod.step, start_prod.save = 0, 0
    
    stack, step = [start_prod], 0
    while stack:
        step += 1
        if step > max_steps: return []
        prod = stack.pop()

        lv = prod._find_first_unexpanded_leaf()

        if not lv:
            prod.update_height()
            if min_depth <= prod.height <= max_depth and \
               (skip_check or all(x.check() for x in [prod] + list(prod.descendants))):
                return [prod]
            continue

        current_target_min_depth = lv.state.get('target_min_depth', 0)
        if debug: print(f"[{step}] Expanding '{lv.type}' @ d={lv.depth} (goal_min={current_target_min_depth}) | Stack: {len(stack)}")
            
        # --- RULE SELECTION AND PRUNING ---
        potential_rules = Rule.get_rules(lv.type, shuffle=True)
        valid_non_terminals, valid_terminals = [], []

        for r in potential_rules:
            # Min height from this rule to a leaf
            min_h_rule = 0 if not r.args else 1 + max((min_heights.get(a, max_depth + 1) for a in r.args), default=0)
            # Max height this rule could possibly produce
            max_h_rule = 0 if not r.args else 1 + max((max_heights.get(a, 0) for a in r.args), default=0)

            # Hard Pruning:
            # 1. Don't exceed max_depth
            if lv.depth + min_h_rule > max_depth: continue
            # 2. Must be able to reach the target depth for this branch
            if lv.depth + max_h_rule < current_target_min_depth: continue
            
            if r.args:
                valid_non_terminals.append(r)
            else: # Terminal rule
                # 3. If we pick a terminal, this leaf's depth must meet the target
                if lv.depth >= current_target_min_depth:
                    valid_terminals.append(r)
        
        # --- DECIDE WHETHER TO TERMINATE OR CONTINUE ---
        rules_to_consider = []
        # If the branch has met its depth goal and we can terminate, we should
        # prefer to do so, especially with low bushiness.
        can_terminate = len(valid_terminals) > 0
        has_met_goal = lv.depth >= current_target_min_depth

        # With probability (1-bushiness), force termination if possible.
        # This creates "vines" for low bushiness.
        if has_met_goal and can_terminate and random.random() > bushiness:
             rules_to_consider = valid_terminals
        else:
             rules_to_consider = valid_non_terminals + valid_terminals

        if debug: print(f"  Candidates ({len(rules_to_consider)}): {[r.name for r in rules_to_consider]}")
        if not rules_to_consider: continue
            
        rules_to_try = s_choices(rules_to_consider, [r.weight for r in rules_to_consider], k)
        if len(rules_to_try) > 1 and random.random() < save_prob: save(prod, stack, step)
        
        for i, rule in enumerate(rules_to_try):
            branch = prod if i == len(rules_to_try) - 1 else prod.clone()
            
            path, curr = [], lv
            while curr.parent: path.append(curr.parent.children.index(curr)); curr = curr.parent
            target_lv = branch
            for idx in reversed(path): target_lv = target_lv.children[idx]

            target_lv.rule = rule
            if rule.state: target_lv.local_state.update(rule.state)

            if rule.args:
                # One child MUST carry the torch for the depth requirement.
                critical_candidates = [
                    idx for idx, arg_type in enumerate(rule.args)
                    if target_lv.depth + 1 + max_heights.get(arg_type, 0) >= current_target_min_depth
                ]
                if not critical_candidates: continue # Should not happen due to prior checks
                
                critical_idx = random.choice(critical_candidates)
                
                new_children = []
                for idx, arg_type in enumerate(rule.args):
                    child = FastProduction(type=arg_type, parent=target_lv)
                    if idx == critical_idx:
                        # This child inherits the same absolute depth target
                        child.local_state['target_min_depth'] = current_target_min_depth
                    else:
                        # Other children get a "bushiness"-controlled target.
                        # For b=0, target is own depth (shallow). For b=1, target is the main target (bushy).
                        remaining_depth = max(0, current_target_min_depth - child.depth)
                        bushy_addon = int(bushiness * remaining_depth)
                        child.local_state['target_min_depth'] = child.depth + random.randint(0, bushy_addon)
                    new_children.append(child)
                target_lv.children = new_children
            else:
                 target_lv.children = []


            if not skip_check and (not target_lv.check('state') or (all(s.rule for s in target_lv.siblings) and not target_lv.parent.check('args'))):
                continue
            
            stack.append(branch)
            break
            
    return []