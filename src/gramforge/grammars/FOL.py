from .. import Substitution, Constraint, generate, init_grammar
import exrex
import funcy as fc
import itertools

C, S = Constraint, Substitution

eng, tptp = "eng","tptp"

ADJECTIVES = ['rich', 'quiet', 'old', 'tall', 'kind', 'brave', 'wise',
              'happy', 'strong', 'curious', 'patient', 'funny', 'generous', 'humble']

NAMES = ['mary', 'paul', 'fred', 'alice', 'john', 'susan', 'lucy']

ADJS=['rich','quiet','old','tall','kind','brave','wise','happy','creative',
    'strong','curious','patient','funny','organized','formal','humble']
ADJS+=['left_handed','curly_haired','popular','romantic','blue_eyed','long_haired','scarred','colorblind']
    

def FOL_grammar(N_PREMS=8, names=NAMES, adjs=ADJS, empty_room=True):
    
    R=init_grammar(['tptp','eng'])
    R('start(setup)', '0')
    
    def render_setup(x):
        persons=x.state['persons']
        room=x.state['room']
        persons_str = ', '.join([p.title() for p in persons])
        eng_setup=f"{persons_str} are the only persons in the {room}.\n{x[0]@eng}"
        if len(persons)==1:
            eng_setup=eng_setup.replace(' are ',' is ').replace('persons','person')
        in_room="&".join([f"room({x})" for x in persons])
        disj="|".join([f"X='{x}'" for x in persons])

        dist = ('&'.join('({}!={})'.format(*a) for a in itertools.product(persons,persons) if len(set(a))!=1))
        tptp_setup=in_room + f'&(dist)' + f"&(![X]:(room(X)=>({disj})))&\n{x[0]@tptp}" 
        return tptp_setup,eng_setup
    
    room = 'room'
    
    def pluralize(s):
        s=s.replace('does','do')
        s=s.replace('is a','are')
        s=s.replace('is ','are ')
        s=s.replace('es ','e ')
        s=s.replace('person','persons')
        return s

        
    psums = [list(names[:i]) for i in range(1, int(len(names)*0.8))]
    
    for persons in psums:
        R('setup(valid_block)', lambda x:render_setup(x)[0], lambda x:render_setup(x)[1],
            vars=dict(persons=persons,room=room))
    
    def rs(x):
        return f"(there_is_a_room)&\n{x[0]@tptp}", f"there is a room.\n{x[0]@eng}"

    if empty_room:
        for persons in [names[:4],[],[]]:
            R('setup(valid_block)', lambda x:rs(x)[0], lambda x:rs(x)[1],
                vars=dict(persons=[],room=room), weight=len(psums))
    
    def neg_constraint(x):
        """avoids material negation issues"""
        for d in x.descendants:
            if type(d.rule)==str:
                continue
            tptp=str(d.rule.templates['tptp'])
            if "=>" in tptp or "<=" in tptp:
                return False
            eng=str(d.rule.templates['eng'])
            if 'who' in eng:
                return False
        return True

    def no_free_var(x):
        s=x[0]@tptp
        return 'X' not in s or set('?!')&set(s)
    R('valid_block(block)','0', constraint=no_free_var) #🏁
    
    nesting_limit = lambda x: x[0].render('eng').count('it is ') <= 2
    R("term(term)", "~(0)", "it is not the case that “0”", constraint=nesting_limit,
      state_constraint=neg_constraint,weight=1)
    
    R("term(term)", "0", "it is true that “0”",constraint=nesting_limit,weight=0.25)
    
    
    R("term(term)", "(mary_dream=>(0))", "Mary dreamt that “0”",constraint=nesting_limit,weight=0.15)    

    
    #R("block(term)", "0", "0",weight=1)
    
    def block_cond(x):
        for d in x.descendants:
            if type(d.rule)!=str and d.rule.name in ['cproperty','term']:
                if '=' in str(d.rule.templates['tptp']):
                    return False
                if 'who' in str(d.rule.templates['eng']):
                    return False
        return True
    
    R('nterm(term)','0', state_constraint=block_cond)
    
    if True:
        
        R("cond(nterm,term)", "(0)=>(1)", "if “0” then “1”", weight=4)
    
        #R("cond(nterm,nterm,term)", "((0)=>(1))&\n((1)=>(2))", "if “0” then “1”\nif “1” then “2”")
        R("cond(nterm,term)", "(0)=>(1)", "“1” if “0”")
        R("cond(nterm,nterm)", "(0)<=>(1)", "“1” if “0” and vice versa")
        R("cond(nterm,term)", "(0)=>(1)", "“0” only if “1”")
        R("cond(nterm,nterm)", "(0)<~>(1)", "either “0” or “1” but not both")
        R("cond(term,nterm)", "~(1)=>(0)", "“0” unless “1”")
        R('cond(nterm,term,term)', '((0)=>(1))&((~(0))=>(2))', 'if “0” then “1” otherwise “2”')
        #R("block(term,term)", "(0)&\n(1)", "0\n1")
    
        conj_tptp = lambda i: '&\n'.join([f'({i})' for i in range(i)])
        conj_eng  = lambda i: '\n'.join([f'{i}' for i in range(i)])
        conj = lambda i: (conj_tptp(i), conj_eng(i))
    
        
        for n in [N_PREMS]:
            for nconds in range(min(10,int(n//2))): #1.5 
                nterms = n - nconds  
                arg1 = "block(" + "cond,"*nconds + ("term,"*(nterms-1)).rstrip(',') + ")"
                w=(2 if nconds==0 else 1)
                w=w*(0.15 if n<=8 else 1)
                R(arg1, *conj(n-1),weight=w)

        
    props = ['propositiona', 'propositionb', 'propositionc','propositiond','propositione']
    props=[]
    for p in props:
        R('prop',p)
        R('prop',"~"+p)
    
    R('term(prop)','0')
    R('term(prop,prop)','((0)&(1))','“0” and “1”')
    R('term(prop,prop,prop)','((0)&(1)&(2))','“0” and “1” and “2”')
    R('term(prop,prop)','((0)|(1))','“0” or “1” or both')
    R('term(prop,prop)','((0)<~>(1))','“0” or “1” but not both')
    R('term(prop,prop)','~((0)|(1))','neither “0” nor “1”')
    
    for n in names:
        R("entity_",n,n.title())
    
    def inside_person(x):
        persons =x[0].state.get('persons','')
        outcome= x[0]@tptp in persons
        return outcome
    
    entity_constraint = None
    R('entity(entity_)','0','0')#,constraint=entity_constraint)
    

    for adj in adjs:
        if adj=="rich":
            continue
        R('adjective', adj,weight=0.8)
    
    R('noun','person')
    R('adj_noun(adjective_chain,noun)','0&1(?)','0 1')
    R('adjective_chain(adjective_chain,adjective)','0&1(?)','0 1',constraint=C('1∉0'))
    R('adjective_chain(adjective)','0(?)','0')
    
    #R('outside_entity(entity_)','0','0',constraint=lambda x: not inside_person(x))
    
    R('property(adj_noun)','0','is a 0',weight=2)
    R('property(adj_noun)','~(0)','is not a 0',weight=1)
    
    R('neg_adj(neg_adj,adjective)','0&~1(?)','0, not 1',constraint=C('1∉0'))
    R('neg_adj(adjective)','~0(?)','not 0')
    R('property(neg_adj)','0','is 0',weight=0.5)
    
    
    R('mneg(mneg)','~0', 'not 0',weight=0.75)
    R('mneg(adjective)','0(?)','0')
    R('property(mneg)','0','is 0',weight=0.5)
    
    
    R('property(adjective)','0(?)','is 0')
    
    R('property(sentiment,outside_entity)', '~0(1,?)','does not 0 1',weight=0.5)
    R('property(sentiment,outside_entity)', '~0(?,1)','is not 0d by 1',weight=0.5)
    # /negations
    
    R('term(entity,entity,adjective,adjective)','(2(0))&(3(1))','0 and 1 are respectively 2 and 3')  
    

    R("sentiment", "like"), R("sentiment", "hate")
    R('property(sentiment,outside_entity)', '0(1,?)','0s 1',weight=0.25)
    R('property(sentiment,outside_entity)', '0(?,1)','is 0d by 1',weight=0.25)
    R('term(entity,outside_entity,sentiment)','(2(0,1) & 2(1,0))', '0 and 1 2 each other')
    R('term(outside_entity,entity,sentiment)','(2(0,1) & 2(1,0))', '0 and 1 2 each other')
    
    R('relation(sentiment)','0'), R('relation(predicate)','0')
    R('term(property,property,relation)',r'\?[X,Y]:((0[?←X])&(1[?←Y])&(2(X,Y)))','someone who 0 2 someone who 1')
    # other quantifiers are ambiguous
    
    preds = list(exrex.generate('pred[a-j]'))

    for i in preds:
        R('atomic_predicate',i)
    R('property(atomic_predicate)', '0(?)','0',weight=6)
    R('property(atomic_predicate)', '~0(?)','~0',weight=1)
    
    R('neg_property(atomic_predicate)', '~0(?)','~0')
    
    R('term(entity,property)', '1[?←0]',
      fc.rcompose(S('0 1', lang='eng'), lambda x: x.replace('they', 'he/she')),
    constraint=[C('0∉1,1∉0'), lambda x: 'who' not in x[1]@eng],
      weight=8)

    R('term(entity,property,property)','(1[?←0])&(2[?←0])','0 who 1 2',weight=1)
    
    R('term(property,entity,entity)', '0[?←1]&0[?←2]', fc.rcompose(S('1 and 2 0',lang='eng'),pluralize),
        constraint=[
            C('0∉1,1∉2,0∉2'),
            lambda x:'pred' not in x[0]@tptp,
            lambda x:'who' not in x[0]@eng],
      weight=2)
    
    R('quantifier', '!', 'everyone',weight=10), 
    R('quantifier','~!','not everyone', state_constraint=neg_constraint)
    ####################R('quantifier','~\?','no one', state_constraint=neg_constraint) 
    R('group','room','in the room',weight=8), R('group','~room','outside the room')
    R('group','anywhere','anywhere')
    
    R('X_quantifier(quantifier,group)','0[X]:(1(X)=>(?))','0 1')
    
    
    R('term(adjective,adjective,group)', '![X]:(2(X)=>(0(X)=>1(X)))','all 0 persons 2 are 1',weight=2)
    R('term(adjective,adjective,group)', '![X]:(2(X)=>(0(X)=>~1(X)))','no 0 person 2 is 1')
    
    def block_property(x):
        for d in x.descendants:
            if d.rule and 'property' in d.rule.args:
                return False
        return True
    
    kw=dict(weight=0.3,state_constraint=block_property)
    R('property(property,property)', '((0)<~>(1))', 'either 0 or 1 but not both',**kw)
    R('property(property,property)', '((0)|(1))', '0 or 1 or both',**kw)
    R('property(property,property)', '((0)&(1))', '0 and 1',**kw)
    R('property(property,property)', '~((0)|(1))', 'neither 0 nor 1',**kw)
    R('property(property,property,property)', '((0)&(1)&(2))', '0, 1 and 2',**kw)
    R('property(property,property,property)', '((0)|(1)|(2))', '0, 1 or 2',**kw)
    
    R('cproperty(property,property)', '((0)=>(1))', 'who 0 1',weight=4)
    R('cproperty(property,property)', '((0)=>(1))', '1 if they 0',weight=4)
    R('cproperty(property,property)', '((1)<=(0))', '0 only if they 1')
    R('cproperty(property,property)', '((0)<=>(1))', '0 if they 1 and vice versa',weight=0.5)
    R('cproperty(property,property)', '((0)<=>(1))', '0 if and only if they 1',weight=0.5)
    #unless, otherwise
    
    #implication as disjunction for hypothesis (more explicit)
    R('cproperty(neg_property,property)', '((0)|(1))', '1 or 0',weight=0.5,state_constraint=block_property) 
    R('cproperty(property,neg_property)', '((0)|(1))', '1 or 0',weight=0.5,state_constraint=block_property) 
    
    
    R('X_property(cproperty)','0[?←X]','0')
    R('X_property(property)','0[?←X]','0',weight=0.1)
    
    R('term(X_quantifier,X_property)', '0[?←1]', '0 1', weight=12) 

    R('E_quantifier(group)',r'\?[X]:(0(X)&(?))','someone 0')
    R('E_property(property)','0[?←X]','0')
    R('term(E_quantifier,E_property)', '0[?←1]', '0 1', weight=1) 
    
    R('xp(property)','0[?←X]','0')
    R('term(xp,xp)', '(![X]:((0)=>(1)))', 'if someone 0 then he/she 1')
    R('term(xp,xp)', '(![X]:((0)<=>(1)))', 'if someone 0 then he/she 1 and vice versa')
    
    ws=0.5
    R('term(property,group)',r'((\?[X]:(1(X)&0[?←X])))','at least one person 1 0',weight=ws)
    R('term(property,group)',r'((\?[X]:(1(X)&0[?←X]))&(![X,Y]:((1(X)&1(Y)&(0[?←X])&(0[?←Y]))=>(X=Y))))','only one person 1 0',weight=ws)
    R('term(property,group)',r'(\?[X,Y]:(1(X)&1(Y)&(0[?←X]&0[?←Y])&(X!=Y)))','more than one person 1 0',weight=ws)
    

    def at_least(n,cond, prop):
        assert n<25 # relies on letters variables
        variables = [chr(ord('A')+i) for i in range(n)]
        conjuncts = [f'{prop}[?←{var}]' for var in variables]
        inequalities = [f'({variables[i]}!={variables[j]})' for i in range(n) for j in range(i+1, n)]
        conds = [f'{cond}({var})' for var in variables]
        formula = '&'.join(conds+inequalities+conjuncts)
        return f'?[{",".join(variables)}]:({formula})'
    
    def at_most(n,cond, prop):
        return f"~({at_least(n+1,cond,prop)})"
    
    def exactly(n,cond, prop):
        return f"({at_least(n,cond,prop)})&({at_most(n,cond,prop)})"
    
    ws = 0.5/10
    for exp in ['at_least','at_most','exactly']:
        for i in range(2,5+1):
            n={1:'one',2:'two',3:'three',4:'four',5:'five'}[i]
            R('term(property,group)',eval(exp)(i,1,0), f'{exp.replace("_"," ")} {n} persons 1 0',weight=ws)


    def hyp_cst(x):
        for d in x.descendants:
            if type(d.rule)==str:
                continue
            tptp=str(d.rule.templates['tptp'])
            if "=>" in tptp or "<=" in tptp:
                return False
            if 'room' in tptp:
                return False
        return True
    
    R('hypothesis(term)', '0', '0',  state_constraint=[neg_constraint, hyp_cst],
      vars=dict(persons=persons,room=room))


    
    return R