"""Small first-order parser for the emitted, function-free TPTP formulas.

This is independent of the grammar renderer. It rejects free variables and
inconsistent symbol arities and supports open-domain Z3 verification.
"""
import re
import z3

TOKEN = re.compile(r"\s*(<=>|<~>|=>|<=|!=|[~&|!?():,=\[\]]|'[^']+'|[A-Za-z_$][A-Za-z0-9_$]*)")
PRECEDENCE = {'<=>': 1, '<~>': 1, '=>': 2, '<=': 2, '|': 3, '&': 4}


class Parser:
    def __init__(self, text):
        self.tokens = []
        position = 0
        while position < len(text.rstrip()):
            match = TOKEN.match(text, position)
            if not match:
                raise ValueError(f'Invalid formula at {text[position:position + 60]!r}')
            self.tokens.append(match[1]); position = match.end()
        self.i = 0

    def pop(self, expected=None):
        value = self.tokens[self.i]
        self.i += 1
        if expected is not None and value != expected:
            raise ValueError(f'Expected {expected}, found {value}')
        return value

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def formula(self, precedence=0):
        left = self.unary()
        while self.peek() in PRECEDENCE and PRECEDENCE[self.peek()] >= precedence:
            op = self.pop()
            left = (op, left, self.formula(PRECEDENCE[op] + 1))
        return left

    def unary(self):
        if self.peek() == '~':
            self.pop(); return ('~', self.unary())
        if self.peek() in ('!', '?'):
            quantifier = self.pop(); self.pop('[')
            variables = [self.pop()]
            while self.peek() == ',':
                self.pop(); variables.append(self.pop())
            self.pop(']'); self.pop(':')
            if any(not v[0].isupper() for v in variables):
                raise ValueError('Quantifiers must bind variables')
            return (quantifier, tuple(variables), self.unary())
        if self.peek() == '(':
            self.pop(); node = self.formula(); self.pop(')'); return node
        name = self.pop().strip("'")
        if self.peek() in ('=', '!='):
            return (self.pop(), name, self.pop().strip("'"))
        terms = []
        if self.peek() == '(':
            self.pop(); terms.append(self.pop().strip("'"))
            while self.peek() == ',':
                self.pop(); terms.append(self.pop().strip("'"))
            self.pop(')')
        return ('atom', name, tuple(terms))


def parse(text):
    parser = Parser(text)
    result = parser.formula()
    if parser.peek() is not None:
        raise ValueError(f'Unconsumed tokens: {parser.tokens[parser.i:]}')
    symbols(result)  # Also checks variable binding and arities.
    return result


def symbols(tree, bound=frozenset(), result=None):
    result = {} if result is None else result
    op = tree[0]
    if op == 'atom':
        _, name, terms = tree
        if name in result and result[name] != len(terms):
            raise ValueError(f'Conflicting arity for {name}')
        result[name] = len(terms)
    elif op in ('=', '!='):
        terms = tree[1:]
    elif op in ('!', '?'):
        return symbols(tree[2], bound | set(tree[1]), result)
    else:
        for child in tree[1:]:
            symbols(child, bound, result)
        return result
    for term in terms:
        if term[0].isupper() and term not in bound:
            raise ValueError(f'Free variable: {term}')
    return result


class Theory:
    def __init__(self):
        self.domain = z3.DeclareSort('Person')
        self.predicates = {}
        self.constants = {}

    def term(self, name, bound):
        if name[0].isupper():
            return bound[name]
        if name not in self.constants:
            self.constants[name] = z3.Const(name, self.domain)
        return self.constants[name]

    def expression(self, node, bound=None):
        bound = {} if bound is None else bound
        op = node[0]
        if op == 'atom':
            _, name, terms = node
            key = name, len(terms)
            if key not in self.predicates:
                self.predicates[key] = z3.Function(name, *([self.domain] * len(terms)), z3.BoolSort()) if terms else z3.Bool(name)
            predicate = self.predicates[key]
            return predicate(*[self.term(t, bound) for t in terms]) if terms else predicate
        if op in ('=', '!='):
            equality = self.term(node[1], bound) == self.term(node[2], bound)
            return equality if op == '=' else z3.Not(equality)
        if op in ('!', '?'):
            variables = {v: z3.Const(v, self.domain) for v in node[1]}
            body = self.expression(node[2], bound | variables)
            return (z3.ForAll if op == '!' else z3.Exists)(list(variables.values()), body)
        if op == '~':
            return z3.Not(self.expression(node[1], bound))
        a, b = [self.expression(n, bound) for n in node[1:]]
        return {'&': lambda: z3.And(a, b), '|': lambda: z3.Or(a, b),
                '=>': lambda: z3.Implies(a, b), '<=': lambda: z3.Implies(b, a),
                '<=>': lambda: a == b, '<~>': lambda: z3.Xor(a, b)}[op]()


def z3_check(formulas, timeout_ms=3000):
    theory = Theory()
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    arities = {}
    for formula in formulas:
        tree = parse(formula)
        for name, arity in symbols(tree).items():
            if name in arities and arities[name] != arity:
                raise ValueError(f'Conflicting arity for {name}')
            arities[name] = arity
        solver.add(theory.expression(tree))
    status = solver.check()
    return str(status)
