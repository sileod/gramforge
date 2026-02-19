from .generate import generate
from .generate_choices import generate_with_choices, generate_choices_batch
from .grammar import Substitution, init_grammar, Constraint


import nltk
import re

def gramforge_to_nltk(g, lang_index=0):
    """
    Converts a gramforge Grammar class to an nltk.CFG object.
    """
    lang = g.langs[lang_index]
    prods = []

    for r in g._instances:
        lhs = nltk.Nonterminal(r.name)
        tmpl = r.templates[lang]
        rhs = []

        # 1. Terminals (No arguments)
        if not r.args:
            # If dynamic (lambda), call once to get a static repr; else use string
            val = tmpl() if callable(tmpl) else tmpl
            # Handle epsilon: empty string -> empty rhs (NLTK supports this)
            if str(val).strip():
                rhs.append(str(val).strip())
            # else: rhs stays empty for epsilon production

        # 2. Non-Terminals with String Templates (e.g., "{0} + {1}")
        elif isinstance(tmpl, str):
            # Split by placeholder pattern {0}, {1}...
            for part in re.split(r'\{(\d+)\}', tmpl):
                if part.isdigit():
                    # Map index '0' -> NonTerminal(r.args[0])
                    rhs.append(nltk.Nonterminal(r.args[int(part)]))
                elif part.strip():
                    # Add literal string parts as terminals
                    rhs.append(part.strip())

        # 3. Fallback for complex callables (Just sequence the args)
        else:
            rhs = [nltk.Nonterminal(a) for a in r.args]

        prods.append(nltk.grammar.Production(lhs, rhs))

    return nltk.CFG(nltk.Nonterminal(g.start().name), prods)

# Backward-compatible alias
unigram_to_nltk = gramforge_to_nltk