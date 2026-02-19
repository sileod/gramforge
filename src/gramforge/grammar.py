
import os, sys, re, random
import pandas as pd
from dataclasses import dataclass
import itertools
from functools import lru_cache, partial
from tqdm.auto import tqdm
from typing import Any
from anytree import Node, NodeMixin, LightNodeMixin, RenderTree
import psutil
from tqdm.auto import tqdm
import copy
from collections import defaultdict
import numpy as np
from easydict import EasyDict as edict
from functools import wraps
import time
from functools import lru_cache
from collections import deque
import random

class FlatList(list):
    def __iadd__(self, other):
        if isinstance(other, list):
            self.extend(other)
        else:
            self.append(other)
        return self

    def __add__(self, other):
        result = FlatList(self)
        if isinstance(other, list):
            result.extend(other)
        else:
            result.append(other)
        return result


def Constraint(constraint_str):
    def generated_function(x):
        conditions = constraint_str.split(',')
        for cond in conditions:
            i, j = map(int, cond.split('∉'))
            if x[i].render('eng') in x[j].render('eng'):
                return False
        return True
    return generated_function

def apply_to_all_args(f):
    def decorator(func):
        def wrapper(*args, **kwargs):
            new_args = [f(arg) for arg in args]
            return func(*new_args, **kwargs)
        return wrapper
    return decorator


def Substitution(template, lang=None):
    def replace_template(template, a):
        # Make numbers formattable 0 -> {0}
        wrap = lambda s: (re.sub(r'(\d+)', r'{\1}', s) if isinstance(s, str) else s)

        # Function to safely replace only unescaped '?'
        def replace_match(m):
            slot_idx = int(m.group(1))
            replacement = m.group(2)
            # The children args 'a' might be FastProduction nodes, so render them if needed
            arg_to_sub = a[slot_idx]
            if not isinstance(arg_to_sub, str):
                 arg_to_sub = arg_to_sub.render(lang)
            return re.sub(r'(?<!\\)\?', replacement, arg_to_sub)

        inner_replaced = re.sub(r"(\d+)\[\?←(.+?)\]", replace_match, template)
        
        # Render the remaining children before formatting
        rendered_args = [arg.render(lang) if not isinstance(arg, str) else arg for arg in a]
        output = wrap(inner_replaced).format(*rendered_args)

        # Convert escaped "\?" back to "?" after processing
        return output.replace(r'\?', '?')

    # The original Substitution returned a function that expected rendered strings.
    # The fix here is to make it robust enough to handle node objects too.
    def sub(*a, **ka):
        return replace_template(template, a)

    return sub

# Pre-compile regex patterns
NUMBER_PATTERN = re.compile(r'(\d+)')
SUBSTITUTION_PATTERN = re.compile(r"(\d+)\[\?←(.+?)\]")


default_preprocess_template = lambda s: (re.sub(r'(\d+)', r'{\1}', s) if type(s)==str and '←' not in s else s)

def init_grammar(langs, name='', preprocess_template=default_preprocess_template):
    class Rule:
        _instances = []
    
        @classmethod
        def init(cls, langs, name='', preprocess_template=preprocess_template):
            Rule.langs = langs
            Rule._instances = []
            Rule.preprocess_template = preprocess_template
            return langs
    
        def __init__(self, signature, *args, constraint=[], state_constraint=[], vars=dict(), weight=1):
            self.signature=signature
            self.name, self.args = self.parse_signature(signature)
            self.constraint = FlatList() + constraint
            self.state_constraint = FlatList() + state_constraint
            self.templates = {}
            self.weight=weight
            self.state = vars
            for lang, template in zip(Rule.langs, args*2):
                self.templates[lang] = Rule.preprocess_template(template)
            self.index = len(Rule._instances)
            Rule._instances.append(self)
    
        #def __getattr__(self,lang):
        #    return self.templates[lang]
    
        def parse_signature(self, s):
            name, args = re.match(r'(\w+)(?:\((.*?)\))?', s).groups()
            args = args.split(',') if args else []
            args = [a.strip() for a in args]
            return name, args
    
        @classmethod
        def get_rules(cls, name, sort=False, terminals=False,shuffle=False):
            instances = [x for x in cls._instances if x.name == name]
            if terminals:
                instances = [x for x in instances if not x.args]
            if shuffle:
                random.shuffle(instances)
            return instances
    
        @classmethod
        def start(cls):
            starts = cls.get_rules('start')
            if starts:
                return starts[0]
            else:
                return cls._instances[0]
    
        def __repr__(self):
            return f"RULE:{self.name}{self.args}"



        def __reduce__(self):
            def rebuild_rule(signature, constraint, state_constraint, state, weight, templates):
                # Logic to recreate the rule
                rule = Rule(signature, constraint=constraint, state_constraint=state_constraint, 
                            vars=state, weight=weight)
                rule.templates = templates
                return rule
                    
            return (rebuild_rule, (self.signature, self.constraint, self.state_constraint, 
                                  self.state, self.weight, self.templates))
                    
    R = Rule
    R.init(langs,name)

    classname = f'Rule_{id(object())}'
    cls_dict = dict(Rule.__dict__)
    cls_dict.pop('__dict__', None)
    cls_dict.pop('__weakref__', None)
    NewRule = type(classname, (object,), cls_dict)
    globals()[classname] = NewRule
    module = sys.modules[__name__]
    setattr(module, classname, NewRule)
    return NewRule

    