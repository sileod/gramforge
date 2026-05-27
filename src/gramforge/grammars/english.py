from gramforge import init_grammar


def _distinct_slots(spec, lang='eng'):
    coref = {'him': 'he', 'her': 'she', 'us': 'we', 'them': 'they'}

    def toks(node):
        return [coref.get(t, t) for t in node.render(lang).split()]

    def f(x):
        for cond in spec.split(','):
            i, j = map(int, cond.split('∉'))
            if i >= len(x) or j >= len(x):
                continue
            a, b = toks(x[i]), toks(x[j])
            if any(b[k:k + len(a)] == a for k in range(len(b) - len(a) + 1)):
                return False
        return True
    return f


def simple_english_grammar(cap=3, questions=True):
    R = init_grammar(['eng'], preprocess_template=lambda s: s)

    R('start(root)', "{0}")
    R('root(decl)', '{0}.')
    R('root(discourse)', '{0}.')
    R('root(question)', '{0}?')

    # --- Lists ---
    # Split nouns by start-sound
    humans_c = [('scientist', 'scientists'), ('student', 'students'),
                ('teacher', 'teachers'), ('friend', 'friends')]
    humans_v = [('artist', 'artists'), ('engineer', 'engineers')]
    animals_c = [('cat', 'cats'), ('dog', 'dogs')]
    things_c = [('book', 'books'), ('gift', 'gifts'), ('message', 'messages')]
    things_v = [('idea', 'ideas')]
    loc_groups = {
        'in': [('room', 'rooms'), ('school', 'schools'), ('garden', 'gardens')],
        'on': [('table', 'tables'), ('desk', 'desks'), ('bridge', 'bridges')],
        'under': [('table', 'tables'), ('desk', 'desks'), ('tree', 'trees')],
    }
    nouns_c = humans_c + animals_c
    nouns_v = humans_v

    def typed_np(prefix, pairs_c, pairs_v):
        n, np = f'n_{prefix}', f'np_{prefix}'
        for s, p in pairs_c: R(f'{n}_sg_c', s); R(f'{n}_pl', p)
        for s, p in pairs_v: R(f'{n}_sg_v', s); R(f'{n}_pl', p)
        R(f'{n}_sg_any({n}_sg_c)', '{0}'); R(f'{n}_sg_any({n}_sg_v)', '{0}')
        for start, adj in [('c', 'adj_c'), ('v', 'adj_v')]:
            R(f'{np}_part_{start}({n}_sg_{start})', '{0}')
            R(f'{np}_part_{start}({adj}, {n}_sg_any)', '{0} {1}', weight=0.4)
            R(f'{np}_base_sg(det_sg_univ, {np}_part_{start})', '{0} {1}')
        R(f'{np}_base_sg(det_sg_a, {np}_part_c)', '{0} {1}')
        R(f'{np}_base_sg(det_sg_an, {np}_part_v)', '{0} {1}')
        R(f'{np}_base_pl(det_pl, opt_adj, {n}_pl)', '{0} {1}{2}')
        R(f'{np}_sg_full({np}_base_sg)', '{0}')
        R(f'{np}_pl_full({np}_base_pl)', '{0}')

    # Adjectives split by start-sound so article choice can stay natural.
    adjs_c = ['happy', 'sad', 'kind', 'quiet', 'brave', 'curious', 'friendly'][:cap]
    adjs_v = ['open', 'honest', 'odd'][:max(1, cap // 2)]

    # --- Sentence Types ---
    R('decl(decl_simple)', '{0}')
    R('decl(decl_simple, conj, decl_simple)', '{0}, {1} {2}', weight=0.2)
    R('discourse(decl)', '{0}')
    R('discourse(decl, conj, decl)', '{0}, {1} {2}', weight=0.1)
    _c_sv = _distinct_slots("0∉1,1∉0")
    R('decl_simple(np_sg_subj, vp_sg)', '{0} {1}', constraint=_c_sv)
    R('decl_simple(np_pl_subj, vp_pl)', '{0} {1}', constraint=_c_sv)
    R('decl_simple(np_sg_agent, vp_sg_agent)', '{0} {1}', constraint=_c_sv, weight=0.6)
    R('decl_simple(np_pl_agent, vp_pl_agent)', '{0} {1}', constraint=_c_sv, weight=0.6)
    R('conj', 'and'); R('conj', 'but'); R('conj', 'yet')
    
    # Existential 'There' (Fixed for a/an) — lower weight so they don't dominate
    R('decl_simple(there, is, det_sg_a, n_sg_c)', '{0} {1} {2} {3}', weight=0.3)
    R('decl_simple(there, is, det_sg_an, n_sg_v)', '{0} {1} {2} {3}', weight=0.3)
    R('decl_simple(there, are, det_pl_indef, n_pl)', '{0} {1} {2} {3}', weight=0.3)
    R('decl_simple(there, is, det_sg_a, n_thing_sg_c)', '{0} {1} {2} {3}', weight=0.3)
    R('decl_simple(there, is, det_sg_an, n_thing_sg_v)', '{0} {1} {2} {3}', weight=0.3)
    R('decl_simple(there, are, det_pl_indef, n_thing_pl)', '{0} {1} {2} {3}', weight=0.3)

    # --- Questions ---
    # Do-support
    if questions:
        R('question(does, np_sg_subj, vp_lex_base)', '{0} {1} {2}')
        R('question(do, np_pl_subj, vp_lex_base)', '{0} {1} {2}')
        R('question(does, np_sg_agent, vp_agent_base)', '{0} {1} {2}', weight=0.6)
        R('question(do, np_pl_agent, vp_agent_base)', '{0} {1} {2}', weight=0.6)
        # Copula
        R('question(is, np_sg_subj, adj)', '{0} {1} {2}')
        R('question(are, np_pl_subj, adj)', '{0} {1} {2}')
        # Wh-Obj/Adv (Do-support)
        R('question(wh_obj, does, np_sg_subj, v_trans_base)', '{0} {1} {2} {3}')
        R('question(wh_obj, do, np_pl_subj, v_trans_base)', '{0} {1} {2} {3}')
        R('question(wh_adv, does, np_sg_subj, v_intr_base)', '{0} {1} {2} {3}')
        R('question(wh_adv, do, np_pl_subj, v_intr_base)', '{0} {1} {2} {3}')
        R('question(wh_reason, is, np_sg_subj, adj)', '{0} {1} {2} {3}')
        R('question(wh_reason, are, np_pl_subj, adj)', '{0} {1} {2} {3}')
        # Wh-Subj (No do-support)
        R('question(who, vp_sg)', '{0} {1}')

    # --- Relative Clauses ---
    R('rel_subj_sg(that, vp_sg)', ' {0} {1}')
    R('rel_subj_pl(that, vp_pl)', ' {0} {1}')
    R('rel_obj(that, np_sg_subj, v_trans_sg)', ' {0} {1} {2}')
    R('rel_obj(that, np_pl_subj, v_trans_base)', ' {0} {1} {2}')
    R('that', 'that')

    # --- Noun Phrases (Fixed for a/an) ---
    
    # 1. Terminals
    for s, p in nouns_c: R('n_sg_c', s); R('n_pl', p)
    for s, p in nouns_v: R('n_sg_v', s); R('n_pl', p)
    typed_np('h', humans_c, humans_v)
    typed_np('thing', things_c, things_v)
    for prep, pairs in loc_groups.items():
        typed_np(f'loc_{prep}', pairs, [])
        R(f'prep_{prep}', prep)
        R(f'PP(prep_{prep}, np_loc_{prep}_sg_full)', '{0} {1}')
        R(f'PP(prep_{prep}, np_loc_{prep}_pl_full)', '{0} {1}')
    R('n_sg_any(n_sg_c)', '{0}'); R('n_sg_any(n_sg_v)', '{0}') # For use with adjectives
    
    for a in adjs_c: R('adj_c', a); R('adj', a)
    for a in adjs_v: R('adj_v', a); R('adj', a)
    for s in ['best', 'worst', 'biggest', 'smallest']: R('sup', s)

    # 2. Determiners
    for d in ['the', 'this', 'that', 'every']: R('det_sg_univ', d) # Works with C and V
    R('det_sg_a', 'a')
    R('det_sg_an', 'an')
    R('the', 'the') # Explicit for superlative
    
    for d in ['the', 'some', 'many', 'these', 'those']: R('det_pl', d)
    R('det_pl_indef', 'some'); R('det_pl_indef', 'many')

    # 3. Phonetic Chunks (Pre-modifier + Noun)
    
    # Consonant Start: "cat", "happy idea", "happy cat"
    R('np_part_c(n_sg_c)', '{0}')
    R('np_part_c(adj_c, n_sg_any)', '{0} {1}', weight=0.4)

    # Vowel Start: "idea", "ancient cat"
    R('np_part_v(n_sg_v)', '{0}')
    R('np_part_v(adj_v, n_sg_any)', '{0} {1}', weight=0.3)

    # Superlative Part: "best cat" (Always consonant start, requires 'the')
    R('np_part_sup(sup, n_sg_any)', '{0} {1}')

    # 4. Base NPs (Det + Chunk)
    # Universal Dets work with anything
    R('np_base_sg(det_sg_univ, np_part_c)', '{0} {1}')
    R('np_base_sg(det_sg_univ, np_part_v)', '{0} {1}')
    # 'A' works only with Consonant starts
    R('np_base_sg(det_sg_a, np_part_c)', '{0} {1}')
    # 'An' works only with Vowel starts
    R('np_base_sg(det_sg_an, np_part_v)', '{0} {1}')
    # Superlatives force 'the'
    R('np_base_sg(the, np_part_sup)', '{0} {1}', weight=0.1)

    # Plural Base
    R('opt_adj', '')
    R('opt_adj(adj_c)', '{0} ', weight=0.4)
    R('opt_adj(adj_v)', '{0} ', weight=0.2)
    R('np_base_pl(det_pl, opt_adj, n_pl)', '{0} {1}{2}')
    R('np_base_pl(the, sup, n_pl)', '{0} {1} {2}', weight=0.1)

    # 5. Full NPs (Base + Post-modifiers like PP/RC)
    # Singular
    R('np_sg_full(np_base_sg)', '{0}')
    R('np_sg_full(np_base_sg, PP)', '{0} {1}', weight=0.2)
    R('np_sg_full(np_base_sg, rel_subj_sg)', '{0}{1}', weight=0.2)
    R('np_sg_full(np_base_sg, rel_obj)', '{0}{1}', weight=0.2)
    # Plural
    R('np_pl_full(np_base_pl)', '{0}')
    R('np_pl_full(np_base_pl, PP)', '{0} {1}', weight=0.2)
    R('np_pl_full(np_base_pl, rel_subj_pl)', '{0}{1}', weight=0.2)
    R('np_pl_full(np_base_pl, rel_obj)', '{0}{1}', weight=0.2)

    # 6. Roles
    # Subject
    R('np_sg_subj(np_sg_full)', '{0}')
    R('np_sg_subj(pro_sg_subj)', '{0}')
    R('np_sg_subj(name)', '{0}')
    R('np_pl_subj(np_pl_full)', '{0}')
    R('np_pl_subj(pro_pl_subj)', '{0}')
    R('np_sg_agent(np_h_sg_full)', '{0}')
    R('np_sg_agent(pro_sg_agent)', '{0}')
    R('np_sg_agent(name)', '{0}')
    R('np_pl_agent(np_h_pl_full)', '{0}')
    R('np_pl_agent(pro_pl_agent)', '{0}')

    # Object
    R('np_sg_obj(np_sg_full)', '{0}')
    R('np_sg_obj(pro_sg_obj)', '{0}')
    R('np_sg_obj(name)', '{0}')
    R('np_pl_obj(np_pl_full)', '{0}')
    R('np_pl_obj(pro_pl_obj)', '{0}')
    R('np_person_obj(np_h_sg_full)', '{0}')
    R('np_person_obj(np_h_pl_full)', '{0}')
    R('np_person_obj(pro_sg_obj)', '{0}')
    R('np_person_obj(pro_pl_obj)', '{0}')
    R('np_person_obj(name)', '{0}')

    # Indirect Object (IO): keep this animate and avoid reflexive-ish "we/us" artifacts.
    R('np_io(pro_io_sg)', '{0}')
    R('np_io(pro_io_pl)', '{0}')
    R('np_io(name)', '{0}')

    # Direct Object (DO) in Double-Object construction:
    # Restrict pronouns. "Give him it" is awkward. "Give him the book" is standard.
    R('np_transfer(np_thing_sg_full)', '{0}')
    R('np_transfer(np_thing_pl_full)', '{0}')

    # --- Verb Phrases (Fixed for Ditransitives) ---
    
    R('opt_adv', '')
    R('opt_adv(adv)', ' {0}', weight=0.4)

    # 1. Intransitive
    R('vp_sg(v_intr_sg, opt_adv)', '{0}{1}')
    R('vp_pl(v_intr_base, opt_adv)', '{0}{1}')
    R('vp_lex_base(v_intr_base, opt_adv)', '{0}{1}')

    # 2. Transitive
    R('np_obj(np_sg_obj)', '{0}'); R('np_obj(np_pl_obj)', '{0}'); R('np_obj(np_transfer)', '{0}')
    for vp, verb in [('vp_sg', 'v_trans_sg'), ('vp_pl', 'v_trans_base'), ('vp_lex_base', 'v_trans_base')]:
        R(f'{vp}({verb}, np_obj)', '{0} {1}')
    for vp, verb in [('vp_sg_agent', 'v_person_trans_sg'),
                     ('vp_pl_agent', 'v_person_trans_base'),
                     ('vp_agent_base', 'v_person_trans_base')]:
        R(f'{vp}({verb}, np_person_obj)', '{0} {1}')

    # 3. Ditransitive
    # Structure A: Double Object (V IO DO) -> "Gives Alice the book"
    R('vp_sg_agent(v_ditrans_sg, np_io, np_transfer)', '{0} {1} {2}')
    R('vp_pl_agent(v_ditrans_base, np_io, np_transfer)', '{0} {1} {2}')
    R('vp_agent_base(v_ditrans_base, np_io, np_transfer)', '{0} {1} {2}')
    
    # Structure B: Prepositional Dative (V DO to IO) -> "Gives the book to Alice"
    # Transferable direct objects avoid "send Bob to her" artifacts.
    R('to', 'to')
    _c_do_io = _distinct_slots("1∉3,3∉1")
    R('vp_sg_agent(v_ditrans_sg, np_transfer, to, np_io)', '{0} {1} {2} {3}', weight=0.5, constraint=_c_do_io)
    R('vp_pl_agent(v_ditrans_base, np_transfer, to, np_io)', '{0} {1} {2} {3}', weight=0.5, constraint=_c_do_io)
    R('vp_agent_base(v_ditrans_base, np_transfer, to, np_io)', '{0} {1} {2} {3}', weight=0.5, constraint=_c_do_io)

    # 4. Copula
    R('vp_sg(is, adj)', '{0} {1}')
    R('vp_pl(are, adj)', '{0} {1}')

    R('PP(prep_rel, np_sg_obj)', '{0} {1}')
    R('PP(prep_rel, np_pl_obj)', '{0} {1}')

    # --- Vocabulary ---
    R('pro_sg_subj', 'he'); R('pro_sg_subj', 'she'); R('pro_sg_subj', 'it')
    R('pro_pl_subj', 'they'); R('pro_pl_subj', 'we')
    R('pro_sg_agent', 'he'); R('pro_sg_agent', 'she')
    R('pro_pl_agent', 'they'); R('pro_pl_agent', 'we')
    R('pro_sg_obj', 'him'); R('pro_sg_obj', 'her'); R('pro_sg_obj', 'it')
    R('pro_pl_obj', 'them'); R('pro_pl_obj', 'us')
    R('pro_io_sg', 'him'); R('pro_io_sg', 'her')
    R('pro_io_pl', 'them')
    
    R('does', 'does'); R('do', 'do')
    R('is', 'is'); R('are', 'are')
    R('there', 'there')
    R('wh_obj', 'what'); R('wh_obj', 'who') 
    R('who', 'who')
    R('wh_adv', 'where'); R('wh_adv', 'when'); R('wh_adv', 'why')
    R('wh_reason', 'why')

    R('name', 'Alice'); R('name', 'Bob'); R('name', 'Charlie')
    
    for a in ['quickly', 'silently', 'rarely', 'suddenly', 'furiously'][:cap]: R('adv', a)
    for p in ['near', 'beside', 'with']: R('prep', p); R('prep_rel', p)

    for base, sg in [('sleep', 'sleeps'), ('run', 'runs'), ('arrive', 'arrives')]:
        R('v_intr_base', base); R('v_intr_sg', sg)
    for base, sg in [('see', 'sees'), ('find', 'finds'), ('like', 'likes')]:
        R('v_trans_base', base); R('v_trans_sg', sg)
    for base, sg in [('meet', 'meets'), ('help', 'helps'), ('love', 'loves'), ('know', 'knows')]:
        R('v_person_trans_base', base); R('v_person_trans_sg', sg)
    for base, sg in [('give','gives'), ('offer','offers'), ('send','sends')]:
        R('v_ditrans_base', base); R('v_ditrans_sg', sg)

    return R
