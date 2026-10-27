"""The upstream FOL grammar, with explicit repairs and restored original families.

No finite-domain closure is introduced. This profile extends the current
FOL factory; stable logical symbols are never lexicalized.
"""
import re

from ..grammars.FOL import FOL_grammar, NAMES
import json
from pathlib import Path

PREDICATES = json.loads((Path(__file__).parent / "vocabulary.json").read_text())["predicates"]

COMPARISONS = [('rich', 'richer', 'poor', 'poorer'), ('tall', 'taller', 'short', 'shorter'),
               ('quiet', 'quieter', 'loud', 'louder'), ('old', 'older', 'young', 'younger')]
STEMS = {'has': 'have', 'enjoys': 'enjoy', 'owns': 'own', 'plays': 'play', 'collects': 'collect',
         'reads': 'read', 'writes': 'write', 'practices': 'practice', 'uses': 'use', 'speaks': 'speak',
         'bakes': 'bake', 'knows': 'know', 'volunteers': 'volunteer'}


def negative(phrase):
    if '#' in phrase:
        return phrase
    for prefix, negated in [('is ', 'is not '), ('can ', 'cannot '), ('does ', 'does not '), ('has lived ', 'has not lived '), ('has visited ', 'has not visited ')]:
        if phrase.startswith(prefix):
            return negated + phrase[len(prefix):]
    verb, rest = phrase.split(' ', 1)
    if verb in ('regularly', 'frequently'):
        return 'does not ' + verb + ' ' + base_phrase(rest)
    return 'does not ' + base_phrase(phrase)


def base_phrase(phrase):
    verb, rest = phrase.split(' ', 1)
    stem = STEMS.get(verb)
    if stem is None:
        stem = verb[:-3] + 'y' if verb.endswith('ies') else verb[:-2] if verb.endswith(('ches', 'shes', 'sses', 'xes', 'oes')) else verb[:-1] if verb.endswith('s') else verb
    return stem + ' ' + rest


def agree(text):
    text = re.sub(r'\bis\b', 'are', text)
    text = re.sub(r'\bdoes\b', 'do', text)
    verbs = STEMS | {v: base_phrase(v + ' placeholder').split()[0] for v in ['hosts','creates','works','makes','maintains','participates','watches','drives','travels','builds','streams','develops','composes','designs','trains','mentors','competes','contributes','goes']}
    for verb, stem in verbs.items():
        text = re.sub(rf'\b{verb}\b', stem, text)
    return text


def pluralize(text):
    text = re.sub(r'\bare an? ', 'are ', agree(text))
    return re.sub(r'\bperson\b', 'people', text)


def pronoun_clause(text):
    if text.startswith('neither is '):
        text = 'is neither ' + text[len('neither is '):]
    if text.startswith('either is '):
        text = 'is either ' + text[len('either is '):]
    return 'they ' + agree(text)


def make_grammar(n_premises=8, include_setup=True):
    # Upstream's block has N_PREMS - 1 children (its setup counted as one).
    R = FOL_grammar(N_PREMS=n_premises + 1, include_propositional=True, include_setup=include_setup)
    for rule in R._instances:
        sig = rule.signature
        if sig == 'X_quantifier(quantifier,group)':
            def restricted(q, group):
                quantifier = q @ 'tptp'
                connective = '&' if quantifier == r'~\?' else '=>'
                return f'{quantifier}[X]:({group @ "tptp"}(X){connective}(?))'
            rule.templates['tptp'] = restricted
        elif sig == 'term(X_quantifier,X_property)':
            # Under 'no one', English 'who A B' is a restriction/conjunction,
            # not the negation of a material implication. Keep this ambiguous
            # cross-product out; all its component families remain available.
            def no_negative_conditional(children):
                quantifier, body = children
                return not ((quantifier @ 'tptp').startswith(r'~\?') and
                            any(op in (body @ 'tptp') for op in ['=>', '<=', '<~>']))
            rule.constraint.append(no_negative_conditional)
        elif sig == 'cproperty(property,property)' and callable(rule.templates['eng']):
            source = rule.templates['tptp']
            if source == '(({0})=>({1}))':
                rule.templates['eng'] = lambda a, b: f'{b @ "eng"} if {pronoun_clause(a @ "eng")}'
            elif source == '(({1})<=({0}))':
                rule.templates['eng'] = lambda a, b: f'{a @ "eng"} only if {pronoun_clause(b @ "eng")}'
            elif source == '(({0})<=>({1}))':
                rule.templates['eng'] = lambda a, b: f'{a @ "eng"} if and only if {pronoun_clause(b @ "eng")}'
        elif sig == 'term(xp,xp)':
            suffix = ' and vice versa' if '<=>' in rule.templates['tptp'] else ''
            rule.templates['eng'] = lambda a, b, suffix=suffix: f'if someone {a @ "eng"} then {pronoun_clause(b @ "eng")}{suffix}'
        elif sig == 'term(property,entity,entity)':
            rule.templates['eng'] = lambda prop, a, b: f'{a @ "eng"} and {b @ "eng"} {pluralize(prop @ "eng")}'
        elif sig == 'term(property,property,relation)':
            def existential_relation(a, b, rel):
                phrase = rel @ 'eng'
                if not phrase.startswith('is '):
                    phrase += 's'
                return f'someone who {a @ "eng"} {phrase} someone who {b @ "eng"}'
            rule.templates['eng'] = existential_relation
        elif sig in ('property(atomic_predicate)', 'neg_property(atomic_predicate)'):
            is_negative = '~' in str(rule.templates['tptp'])
            rule.templates['eng'] = lambda p, neg=is_negative: negative(p @ 'eng') if neg else p @ 'eng'
        elif sig == 'term(property,group)' and callable(rule.templates['eng']):
            # Keep upstream counting wording, but conjugate verb predicates too.
            old = rule.templates['eng']
            rule.templates['eng'] = lambda prop, group, old=old: pluralize(old(prop, group))
        elif rule.name == 'atomic_predicate':
            index = ord(rule.templates['tptp'][-1]) - ord('a')
            rule.templates['eng'] = PREDICATES[index]
        elif rule.name == 'prop':
            atom = rule.templates['tptp'].lstrip('~')
            index = ord(atom[-1]) - ord('a')
            positive = ['the alarm is on', 'the door is open', 'the light is on',
                        'the window is open', 'the bell is ringing'][index]
            rule.templates['eng'] = f'it is not the case that “{positive}”' if rule.templates['tptp'].startswith('~') else positive

    # Restore useful original productions, repairing their meanings rather than
    # substituting a different logic or a finite list of possible individuals.
    for index in range(10, 26):
        R('atomic_predicate', 'pred' + chr(ord('a') + index), PREDICATES[index])
    R('quantifier', r'~\?', 'no one')
    for noun in ['european', 'tourist']:
        R('noun', noun, noun.title() if noun == 'european' else noun)
    for predicate, phrase in [('sibling', 'is a sibling of'), ('client', 'is a client of')]:
        R('predicate', predicate, phrase)
    R('term(entity,comparative,entity)', '1(0,2)', '0 is 1 than 2')
    for adjective, comparative, _, _ in COMPARISONS:
        R('comparative', comparative, comparative)
        R('predicate', comparative, f'is {comparative} than')
        # Subject first everywhere, including the background theory.
        R('term(entity,group)',
          lambda name, group, c=comparative: f'({group @ "tptp"}({name @ "tptp"})&(![X]:(({group @ "tptp"}(X)&X!={name @ "tptp"})=>{c}({name @ "tptp"},X))))',
          lambda name, group, a=adjective: f'{name @ "eng"} is the most {a} {group @ "eng"}', weight=.125)
        R('term(entity,group)',
          lambda name, group, c=comparative: f'({group @ "tptp"}({name @ "tptp"})&(![X]:(({group @ "tptp"}(X)&X!={name @ "tptp"})=>{c}(X,{name @ "tptp"}))))',
          lambda name, group, a=adjective: f'{name @ "eng"} is the least {a} {group @ "eng"}', weight=.125)
    R('property(predicate,outside_entity)', '0(?,1)', '0 1', weight=3)
    R('cond(nterm,nterm,term)', '((0)=>(1))&((1)=>(2))', 'if “0” then “1”\nif “1” then “2”')
    for formula, phrase in [('((0)|(1))', 'At least one'), ('((0)<~>(1))', 'Exactly one'),
                            ('~((0)&(1))', 'At most one'), ('~((0)|(1))', 'None')]:
        R('list(term,term)', formula, f'{phrase} of the following statements is true:\n- 0\n- 1')
    R('block(list)', '0', '0', weight=.25)
    # Context reuse is structural: both languages reference exactly the same
    # previously generated property. No object-id seeding or random rendering.
    R('state', '')
    def previous_units(node):
        for ancestor in node.ancestors:
            if ancestor.type == 'term' and ancestor.parent and ancestor.parent.type == 'block':
                siblings = ancestor.parent.children
                return siblings[:siblings.index(ancestor)]
        return []
    def preceding_consequent(node):
        prop = node.parent
        conditional = prop.parent if prop else None
        if not conditional or conditional.type != 'cproperty' or conditional.children[0] is not prop:
            return None
        for previous in reversed(previous_units(node)):
            candidates = [d for d in previous.descendants if d.type == 'cproperty']
            if candidates and all(d.type != 'state' for d in candidates[-1].children[-1].descendants):
                return candidates[-1].children[-1]
        return None
    def earlier_antecedent(node):
        prop = node.parent
        term = prop.parent if prop else None
        if not term or term.rule.signature != 'term(entity,property)':
            return None
        for previous in previous_units(node):
            for candidate in previous.descendants:
                if candidate.type == 'cproperty' and all(d.type != 'state' for d in candidate.children[0].descendants):
                    return candidate.children[0]
        return None
    def reuse(node, lang, finder):
        previous = finder(node)
        return previous @ lang if previous is not None else '#missing_context'
    # Original notebook weights: antecedent/consequent reuse 12, named anchor 4.
    for finder, weight in [(preceding_consequent, 12), (earlier_antecedent, 4)]:
        R('property(state)', lambda node, finder=finder: reuse(node, 'tptp', finder),
          lambda node, finder=finder: reuse(node, 'eng', finder),
          state_constraint=lambda node: '#missing_context' not in (node @ 'tptp'), weight=weight)
    return R


def background(names=NAMES):
    """Public, explicit person-domain and strict-comparison conventions."""
    axioms = ['![X]:person(X)', '![X]:anywhere(X)',
              '![X]:(european(X)=>person(X))', '![X]:(tourist(X)=>person(X))',
              '![X,Y]:(sibling(X,Y)<=>sibling(Y,X))', '![X]:~sibling(X,X)',
              '![X,Y]:(like(X,Y)=>~hate(X,Y))']
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            axioms.append(f'{a}!={b}')
    for high, greater, low, lesser in COMPARISONS:
        axioms += [f'![X]:~{greater}(X,X)', f'![X]:~{lesser}(X,X)',
                   f'![X,Y]:({greater}(X,Y)<=>{lesser}(Y,X))',
                   f'![X,Y]:({greater}(X,Y)=>~{greater}(Y,X))',
                   f'![X,Y,Z]:(({greater}(X,Y)&{greater}(Y,Z))=>{greater}(X,Z))',
                   f'![X,Y]:(({greater}(X,Y)&{high}(Y))=>{high}(X))',
                   f'![X,Y]:(({lesser}(X,Y)&{low}(Y))=>{low}(X))',
                   f'![X]:({high}(X)=>~{low}(X))']
    return axioms
