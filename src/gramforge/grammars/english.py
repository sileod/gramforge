from gramforge import init_grammar

def simple_english_grammar(cap=3, questions=True):
    R = init_grammar(['eng'], preprocess_template=lambda s: s)

    R('start(root)', "{0}")
    R('root(decl)', '{0}.')
    R('root(question)', '{0}?')

    # --- Lists ---
    # Split nouns by start-sound
    nouns_c = [('cat','cats'), ('dog','dogs'), ('scientist','scientists'),
               ('river','rivers'), ('machine','machines'), ('poem','poems')]
    nouns_v = [('idea','ideas')] 
    
    # Adjectives (All currently start with consonants)
    adjs = ['happy', 'sad', 'green', 'furious', 'silent', 'brave', 'complex'][:cap]

    # --- Sentence Types ---
    R('decl(np_sg_subj, vp_sg)', '{0} {1}')
    R('decl(np_pl_subj, vp_pl)', '{0} {1}')
    R('decl(decl, conj, decl)', '{0}, {1} {2}', weight=0.2)
    R('conj', 'and'); R('conj', 'but'); R('conj', 'yet')
    
    # Existential 'There' (Fixed for a/an)
    R('decl(there, is, det_sg_a, n_sg_c)', '{0} {1} {2} {3}')
    R('decl(there, is, det_sg_an, n_sg_v)', '{0} {1} {2} {3}')
    R('decl(there, are, det_pl_indef, n_pl)', '{0} {1} {2} {3}')

    # --- Questions ---
    # Do-support
    if questions:
        R('question(does, np_sg_subj, vp_lex_base)', '{0} {1} {2}')
        R('question(do, np_pl_subj, vp_lex_base)', '{0} {1} {2}')
        # Copula
        R('question(is, np_sg_subj, adj)', '{0} {1} {2}')
        R('question(are, np_pl_subj, adj)', '{0} {1} {2}')
        # Wh-Obj/Adv (Do-support)
        R('question(wh_obj, does, np_sg_subj, v_trans_base)', '{0} {1} {2} {3}')
        R('question(wh_obj, do, np_pl_subj, v_trans_base)', '{0} {1} {2} {3}')
        R('question(wh_adv, does, np_sg_subj, v_intr_base)', '{0} {1} {2} {3}')
        R('question(wh_adv, do, np_pl_subj, v_intr_base)', '{0} {1} {2} {3}')
        R('question(wh_adv, is, np_sg_subj, adj)', '{0} {1} {2} {3}')
        R('question(wh_adv, are, np_pl_subj, adj)', '{0} {1} {2} {3}')
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
    R('n_sg_any(n_sg_c)', '{0}'); R('n_sg_any(n_sg_v)', '{0}') # For use with adjectives
    
    for a in adjs: R('adj', a)
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
    # Note: All our adjectives are consonant-initial, so "adj + idea" is a consonant start.
    R('np_part_c(n_sg_c)', '{0}')
    R('np_part_c(adj, n_sg_any)', '{0} {1}', weight=0.4) 
    
    # Vowel Start: "idea" (No adjectives allowed here, or it would move to C)
    R('np_part_v(n_sg_v)', '{0}')

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
    R('opt_adj', ''); R('opt_adj(adj)', '{0} ', weight=0.4)
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

    # Object
    R('np_sg_obj(np_sg_full)', '{0}')
    R('np_sg_obj(pro_sg_obj)', '{0}')
    R('np_sg_obj(name)', '{0}')
    R('np_pl_obj(np_pl_full)', '{0}')
    R('np_pl_obj(pro_pl_obj)', '{0}')

    # Indirect Object (IO): Can be any object (pronouns fine: "Give HIM the book")
    R('np_io(np_sg_obj)', '{0}')
    R('np_io(np_pl_obj)', '{0}')

    # Direct Object (DO) in Double-Object construction:
    # Restrict pronouns. "Give him it" is awkward. "Give him the book" is standard.
    R('np_do_doc(np_sg_full)', '{0}')
    R('np_do_doc(np_pl_full)', '{0}')
    R('np_do_doc(name)', '{0}')

    # --- Verb Phrases (Fixed for Ditransitives) ---
    
    R('opt_adv', '')
    R('opt_adv(adv)', ' {0}', weight=0.4)

    # 1. Intransitive
    R('vp_sg(v_intr_sg, opt_adv)', '{0}{1}')
    R('vp_pl(v_intr_base, opt_adv)', '{0}{1}')
    R('vp_lex_base(v_intr_base, opt_adv)', '{0}{1}')

    # 2. Transitive
    R('vp_sg(v_trans_sg, np_sg_obj)', '{0} {1}')
    R('vp_sg(v_trans_sg, np_pl_obj)', '{0} {1}')
    R('vp_pl(v_trans_base, np_sg_obj)', '{0} {1}')
    R('vp_pl(v_trans_base, np_pl_obj)', '{0} {1}')
    R('vp_lex_base(v_trans_base, np_sg_obj)', '{0} {1}')
    R('vp_lex_base(v_trans_base, np_pl_obj)', '{0} {1}')

    # 3. Ditransitive
    # Structure A: Double Object (V IO DO) -> "Gives Alice the book"
    R('vp_sg(v_ditrans_sg, np_io, np_do_doc)', '{0} {1} {2}')
    R('vp_pl(v_ditrans_base, np_io, np_do_doc)', '{0} {1} {2}')
    R('vp_lex_base(v_ditrans_base, np_io, np_do_doc)', '{0} {1} {2}')
    
    # Structure B: Prepositional Dative (V DO to IO) -> "Gives the book to Alice"
    # Note: DO here can be a pronoun ("Give it to Alice")
    R('to', 'to')
    R('vp_sg(v_ditrans_sg, np_sg_obj, to, np_io)', '{0} {1} {2} {3}', weight=0.5)
    R('vp_sg(v_ditrans_sg, np_pl_obj, to, np_io)', '{0} {1} {2} {3}', weight=0.5)
    R('vp_pl(v_ditrans_base, np_sg_obj, to, np_io)', '{0} {1} {2} {3}', weight=0.5)
    R('vp_pl(v_ditrans_base, np_pl_obj, to, np_io)', '{0} {1} {2} {3}', weight=0.5)
    R('vp_lex_base(v_ditrans_base, np_sg_obj, to, np_io)', '{0} {1} {2} {3}', weight=0.5)
    R('vp_lex_base(v_ditrans_base, np_pl_obj, to, np_io)', '{0} {1} {2} {3}', weight=0.5)

    # 4. Copula
    R('vp_sg(is, adj)', '{0} {1}')
    R('vp_pl(are, adj)', '{0} {1}')

    R('PP(prep, np_sg_obj)', '{0} {1}')
    R('PP(prep, np_pl_obj)', '{0} {1}')

    # --- Vocabulary ---
    R('pro_sg_subj', 'he'); R('pro_sg_subj', 'she'); R('pro_sg_subj', 'it')
    R('pro_pl_subj', 'they'); R('pro_pl_subj', 'we')
    R('pro_sg_obj', 'him'); R('pro_sg_obj', 'her'); R('pro_sg_obj', 'it')
    R('pro_pl_obj', 'them'); R('pro_pl_obj', 'us')
    
    R('does', 'does'); R('do', 'do')
    R('is', 'is'); R('are', 'are')
    R('there', 'there')
    R('wh_obj', 'what'); R('wh_obj', 'who') 
    R('who', 'who')
    R('wh_adv', 'where'); R('wh_adv', 'when'); R('wh_adv', 'why')

    R('name', 'Alice'); R('name', 'Bob'); R('name', 'Smith') 
    
    for a in ['quickly', 'silently', 'rarely', 'suddenly', 'furiously'][:cap]: R('adv', a)
    for p in ['in', 'on', 'under', 'near', 'beside']: R('prep', p)

    for base, sg in [('sleep','sleeps'), ('run','runs'), ('vanish','vanishes'), ('exist','exists')]:
        R('v_intr_base', base); R('v_intr_sg', sg)
    for base, sg in [('chase','chases'), ('analyze','analyzes'), ('find','finds'), ('love','loves')]:
        R('v_trans_base', base); R('v_trans_sg', sg)
    for base, sg in [('give','gives'), ('offer','offers'), ('teach','teaches')]:
        R('v_ditrans_base', base); R('v_ditrans_sg', sg)

    return R