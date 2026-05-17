from gramforge import init_grammar, Constraint

def simple_english_grammar(cap=3, questions=True):
    R = init_grammar(['eng'], preprocess_template=lambda s: s)

    R('start(root)', "{0}")
    R('root(decl)', '{0}.')
    R('root(discourse)', '{0}.')
    R('root(question)', '{0}?')

    # --- Lexical lists ---
    # Nouns split by start-sound so a/an stays natural.
    nouns_c = [('cat', 'cats'), ('dog', 'dogs'), ('scientist', 'scientists'),
               ('student', 'students'), ('teacher', 'teachers'), ('friend', 'friends')]
    nouns_v = [('artist', 'artists'), ('engineer', 'engineers'), ('actor', 'actors')]

    adjs_c = ['happy', 'sad', 'kind', 'quiet', 'brave', 'curious', 'friendly'][:cap]
    adjs_v = ['open', 'honest', 'odd', 'angry'][:max(1, cap // 2)]

    # --- Sentence types ---
    # 0∉1,1∉0 prevents same-string subject/predicate (e.g. "Alice loves Alice").
    _c_sv = Constraint("0∉1,1∉0")
    R('decl(decl_simple)', '{0}')
    R('decl(decl_simple, conj, decl_simple)', '{0}, {1} {2}', weight=0.2)
    R('discourse(decl)', '{0}')
    R('discourse(decl, conj, decl)', '{0}, {1} {2}', weight=0.1)
    R('decl_simple(np_sg_subj, vp_sg)', '{0} {1}', constraint=_c_sv)
    R('decl_simple(np_pl_subj, vp_pl)', '{0} {1}', constraint=_c_sv)
    # Past tense (number-invariant for action verbs)
    R('decl_simple(np_sg_subj, vp_past)', '{0} {1}', weight=0.4, constraint=_c_sv)
    R('decl_simple(np_pl_subj, vp_past)', '{0} {1}', weight=0.4, constraint=_c_sv)
    R('decl_simple(np_sg_subj, was, adj)', '{0} {1} {2}', weight=0.2)
    R('decl_simple(np_pl_subj, were, adj)', '{0} {1} {2}', weight=0.2)
    # Negation via do-support (split: "does not" and the "doesn't" contraction)
    _c_sv3 = Constraint("0∉3,3∉0")  # subj vs vp when vp is slot 3
    _c_sv2 = Constraint("0∉2,2∉0")  # subj vs vp when vp is slot 2
    R('decl_simple(np_sg_subj, does, not_, vp_action_base)', '{0} {1} {2} {3}', weight=0.15, constraint=_c_sv3)
    R('decl_simple(np_pl_subj, do, not_, vp_action_base)', '{0} {1} {2} {3}', weight=0.15, constraint=_c_sv3)
    R('decl_simple(np_sg_subj, doesnt, vp_action_base)', '{0} {1} {2}', weight=0.15, constraint=_c_sv2)
    R('decl_simple(np_pl_subj, dont, vp_action_base)', '{0} {1} {2}', weight=0.15, constraint=_c_sv2)
    # Modals (number-invariant)
    R('decl_simple(np_sg_subj, modal, vp_action_base)', '{0} {1} {2}', weight=0.3, constraint=_c_sv2)
    R('decl_simple(np_pl_subj, modal, vp_action_base)', '{0} {1} {2}', weight=0.3, constraint=_c_sv2)
    R('conj', 'and'); R('conj', 'but'); R('conj', 'yet')
    # Subordinating conjunction ("she sleeps because he runs")
    R('decl_simple(decl_simple, sub_conj, decl_simple)', '{0} {1} {2}',
      weight=0.08, constraint=Constraint("0∉2,2∉0"))
    for s in ['because', 'when', 'if', 'although']: R('sub_conj', s)

    # Existential 'there'
    R('decl_simple(there, is, det_sg_a, n_sg_c)', '{0} {1} {2} {3}', weight=0.3)
    R('decl_simple(there, is, det_sg_an, n_sg_v)', '{0} {1} {2} {3}', weight=0.3)
    R('decl_simple(there, are, det_pl_indef, n_pl)', '{0} {1} {2} {3}', weight=0.3)

    # --- Questions ---
    if questions:
        # Do-support (present)
        R('question(does, np_sg_subj, vp_action_base)', '{0} {1} {2}')
        R('question(do, np_pl_subj, vp_action_base)', '{0} {1} {2}')
        # Did-support (past, number-invariant)
        R('question(did, np_sg_subj, vp_action_base)', '{0} {1} {2}')
        R('question(did, np_pl_subj, vp_action_base)', '{0} {1} {2}')
        R('did', 'did')
        # Modal questions ("can she run?")
        R('question(modal, np_sg_subj, vp_action_base)', '{0} {1} {2}', weight=0.5)
        R('question(modal, np_pl_subj, vp_action_base)', '{0} {1} {2}', weight=0.5)
        # Copula present + past
        R('question(is, np_sg_subj, adj)', '{0} {1} {2}')
        R('question(are, np_pl_subj, adj)', '{0} {1} {2}')
        R('question(was, np_sg_subj, adj)', '{0} {1} {2}', weight=0.5)
        R('question(were, np_pl_subj, adj)', '{0} {1} {2}', weight=0.5)
        # Wh-Obj/Adv (do-support, present)
        R('question(wh_obj, does, np_sg_subj, v_trans_base)', '{0} {1} {2} {3}')
        R('question(wh_obj, do, np_pl_subj, v_trans_base)', '{0} {1} {2} {3}')
        R('question(wh_adv, does, np_sg_subj, v_intr_base)', '{0} {1} {2} {3}')
        R('question(wh_adv, do, np_pl_subj, v_intr_base)', '{0} {1} {2} {3}')
        R('question(wh_reason, is, np_sg_subj, adj)', '{0} {1} {2} {3}')
        R('question(wh_reason, are, np_pl_subj, adj)', '{0} {1} {2} {3}')
        # Wh-Subj (no do-support)
        R('question(who, vp_sg)', '{0} {1}')

    # --- Relative clauses ---
    R('rel_subj_sg(that, vp_sg)', ' {0} {1}')
    R('rel_subj_pl(that, vp_pl)', ' {0} {1}')
    R('rel_subj_sg(that, vp_past)', ' {0} {1}', weight=0.5)
    R('rel_subj_pl(that, vp_past)', ' {0} {1}', weight=0.5)
    R('rel_obj(that, np_sg_subj, v_trans_sg)', ' {0} {1} {2}')
    R('rel_obj(that, np_pl_subj, v_trans_base)', ' {0} {1} {2}')
    R('rel_obj(that, np_sg_subj, v_trans_past)', ' {0} {1} {2}', weight=0.5)
    R('rel_obj(that, np_pl_subj, v_trans_past)', ' {0} {1} {2}', weight=0.5)
    R('that', 'that')

    # --- Noun phrases ---
    # 1. Terminals
    for s, p in nouns_c: R('n_sg_c', s); R('n_pl', p)
    for s, p in nouns_v: R('n_sg_v', s); R('n_pl', p)
    R('n_sg_any(n_sg_c)', '{0}'); R('n_sg_any(n_sg_v)', '{0}')

    for a in adjs_c: R('adj_c', a); R('adj', a)
    for a in adjs_v: R('adj_v', a); R('adj', a)
    for s in ['best', 'worst', 'biggest', 'smallest']: R('sup', s)

    # 2. Determiners (possessives like "her" work as both sg and pl universal)
    for d in ['the', 'this', 'that', 'every', 'his', 'her', 'its', 'their', 'our']:
        R('det_sg_univ', d)
    R('det_sg_a', 'a'); R('det_sg_an', 'an'); R('the', 'the')
    for d in ['the', 'some', 'many', 'these', 'those', 'his', 'her', 'their', 'our']:
        R('det_pl', d)
    R('det_pl_indef', 'some'); R('det_pl_indef', 'many')

    # 3. Phonetic chunks (adj + noun)
    R('np_part_c(n_sg_c)', '{0}')
    R('np_part_c(adj_c, n_sg_any)', '{0} {1}', weight=0.4)
    R('np_part_v(n_sg_v)', '{0}')
    R('np_part_v(adj_v, n_sg_any)', '{0} {1}', weight=0.3)
    R('np_part_sup(sup, n_sg_any)', '{0} {1}')

    # 4. Base NPs (det + chunk); a/an stay matched to chunk start-sound
    R('np_base_sg(det_sg_univ, np_part_c)', '{0} {1}')
    R('np_base_sg(det_sg_univ, np_part_v)', '{0} {1}')
    R('np_base_sg(det_sg_a, np_part_c)', '{0} {1}')
    R('np_base_sg(det_sg_an, np_part_v)', '{0} {1}')
    R('np_base_sg(the, np_part_sup)', '{0} {1}', weight=0.1)

    R('opt_adj', '')
    R('opt_adj(adj_c)', '{0} ', weight=0.4)
    R('opt_adj(adj_v)', '{0} ', weight=0.2)
    R('np_base_pl(det_pl, opt_adj, n_pl)', '{0} {1}{2}')
    R('np_base_pl(the, sup, n_pl)', '{0} {1} {2}', weight=0.1)

    # 5. Full NPs (base + optional PP / relative clause)
    R('np_sg_full(np_base_sg)', '{0}')
    R('np_sg_full(np_base_sg, PP)', '{0} {1}', weight=0.2)
    R('np_sg_full(np_base_sg, rel_subj_sg)', '{0}{1}', weight=0.2)
    R('np_sg_full(np_base_sg, rel_obj)', '{0}{1}', weight=0.2)
    R('np_pl_full(np_base_pl)', '{0}')
    R('np_pl_full(np_base_pl, PP)', '{0} {1}', weight=0.2)
    R('np_pl_full(np_base_pl, rel_subj_pl)', '{0}{1}', weight=0.2)
    R('np_pl_full(np_base_pl, rel_obj)', '{0}{1}', weight=0.2)

    # 6. Roles
    # Subject (kept number-marked: verb agreement)
    R('np_sg_subj(np_sg_full)', '{0}')
    R('np_sg_subj(pro_sg_subj)', '{0}')
    R('np_sg_subj(name)', '{0}')
    R('np_pl_subj(np_pl_full)', '{0}')
    R('np_pl_subj(pro_pl_subj)', '{0}')
    # Compound subject ("Alice and Bob") — light weight, blocks same-name pairs
    R('np_pl_subj(np_sg_subj, np_sg_subj)', '{0} and {1}',
      weight=0.1, constraint=Constraint("0∉1,1∉0"))
    # Object (number-unified: verbs don't agree with objects)
    R('np_obj(np_sg_full)', '{0}'); R('np_obj(np_pl_full)', '{0}')
    R('np_obj(pro_sg_obj)', '{0}'); R('np_obj(pro_pl_obj)', '{0}')
    R('np_obj(name)', '{0}')
    # Compound object ("she meets Alice and Bob")
    R('np_obj(np_obj, np_obj)', '{0} and {1}', weight=0.05,
      constraint=Constraint("0∉1,1∉0"))
    # Indirect object (animate)
    R('np_io(pro_io_sg)', '{0}'); R('np_io(pro_io_pl)', '{0}'); R('np_io(name)', '{0}')
    # Direct object in double-object construction (no pronouns: "give him it" is awkward)
    R('np_do_doc(np_sg_full)', '{0}'); R('np_do_doc(np_pl_full)', '{0}')

    # --- Verb phrases ---
    R('opt_adv', '')
    R('opt_adv(adv)', ' {0}', weight=0.4)

    # opt_pp: optional VP-attached PP ("she runs in the park")
    R('opt_pp', '')
    R('opt_pp(PP)', ' {0}', weight=0.2)

    # vp_action_base: bare-form action VP shared by negation, modals, do-questions, vp_pl, vp_lex_base.
    # 1∉3,3∉1 prevents "give Alice to Alice".
    _c_do_io = Constraint("1∉3,3∉1")
    R('vp_action_base(v_intr_base, opt_adv, opt_pp)', '{0}{1}{2}')
    R('vp_action_base(v_trans_base, np_obj)', '{0} {1}')
    R('vp_action_base(v_ditrans_base, np_io, np_do_doc)', '{0} {1} {2}')
    R('vp_action_base(v_ditrans_base, np_obj, to, np_io)', '{0} {1} {2} {3}',
      weight=0.5, constraint=_c_do_io)
    R('vp_pl(vp_action_base)', '{0}')
    R('vp_lex_base(vp_action_base)', '{0}')

    # 3sg present (must inflect, so listed separately from base)
    R('vp_sg(v_intr_sg, opt_adv, opt_pp)', '{0}{1}{2}')
    R('vp_sg(v_trans_sg, np_obj)', '{0} {1}')
    R('vp_sg(v_ditrans_sg, np_io, np_do_doc)', '{0} {1} {2}')
    R('vp_sg(v_ditrans_sg, np_obj, to, np_io)', '{0} {1} {2} {3}',
      weight=0.5, constraint=_c_do_io)

    # Past tense (number-invariant for action verbs)
    R('vp_past(v_intr_past, opt_adv, opt_pp)', '{0}{1}{2}')
    R('vp_past(v_trans_past, np_obj)', '{0} {1}')
    R('vp_past(v_ditrans_past, np_io, np_do_doc)', '{0} {1} {2}')
    R('vp_past(v_ditrans_past, np_obj, to, np_io)', '{0} {1} {2} {3}',
      weight=0.5, constraint=_c_do_io)

    # Copula (present)
    R('vp_sg(is, adj)', '{0} {1}')
    R('vp_pl(are, adj)', '{0} {1}')
    # Bare-form "be" copula (used by modals / negation: "she can be happy")
    R('vp_action_base(be, adj)', '{0} {1}', weight=0.2)
    R('be', 'be')

    # VP coordination ("she runs and sleeps"); same Constraint blocks "X and X"
    _c_vp_coord = Constraint("0∉2,2∉0")
    R('vp_sg(vp_sg, vp_sg)', '{0} and {1}', weight=0.05, constraint=_c_vp_coord)
    R('vp_pl(vp_pl, vp_pl)', '{0} and {1}', weight=0.05, constraint=_c_vp_coord)
    R('vp_past(vp_past, vp_past)', '{0} and {1}', weight=0.05, constraint=_c_vp_coord)
    R('vp_action_base(vp_action_base, vp_action_base)', '{0} and {1}',
      weight=0.04, constraint=_c_vp_coord)

    R('to', 'to')

    R('PP(prep_loc, np_sg_full)', '{0} {1}')
    R('PP(prep_loc, np_pl_full)', '{0} {1}')
    R('PP(prep_rel, np_obj)', '{0} {1}')

    # --- Vocabulary ---
    R('pro_sg_subj', 'he'); R('pro_sg_subj', 'she'); R('pro_sg_subj', 'it')
    R('pro_pl_subj', 'they'); R('pro_pl_subj', 'we')
    R('pro_sg_obj', 'him'); R('pro_sg_obj', 'her'); R('pro_sg_obj', 'it')
    R('pro_pl_obj', 'them'); R('pro_pl_obj', 'us')
    R('pro_io_sg', 'him'); R('pro_io_sg', 'her')
    R('pro_io_pl', 'them')

    R('does', 'does'); R('do', 'do')
    R('is', 'is'); R('are', 'are'); R('was', 'was'); R('were', 'were')
    R('there', 'there')
    R('wh_obj', 'what'); R('wh_obj', 'who'); R('who', 'who')
    R('wh_adv', 'where'); R('wh_adv', 'when'); R('wh_adv', 'why')
    R('wh_reason', 'why')

    R('not_', 'not'); R('doesnt', "doesn't"); R('dont', "don't")
    for m in ['can', 'will', 'should', 'must', 'may', "won't", "shouldn't", "can't"]:
        R('modal', m)

    R('name', 'Alice'); R('name', 'Bob'); R('name', 'Charlie')

    for a in ['quickly', 'silently', 'rarely', 'suddenly', 'furiously'][:cap]: R('adv', a)
    for p in ['in', 'on', 'under']: R('prep', p); R('prep_loc', p)
    for p in ['near', 'beside', 'with']: R('prep', p); R('prep_rel', p)

    for base, sg, past in [('sleep','sleeps','slept'), ('run','runs','ran'),
                           ('arrive','arrives','arrived'), ('exist','exists','existed')]:
        R('v_intr_base', base); R('v_intr_sg', sg); R('v_intr_past', past)
    for base, sg, past in [('meet','meets','met'), ('help','helps','helped'),
                           ('find','finds','found'), ('love','loves','loved'),
                           ('see','sees','saw'), ('like','likes','liked'),
                           ('know','knows','knew'), ('have','has','had'),
                           ('watch','watches','watched'), ('want','wants','wanted'),
                           ('visit','visits','visited'), ('call','calls','called')]:
        R('v_trans_base', base); R('v_trans_sg', sg); R('v_trans_past', past)
    for base, sg, past in [('give','gives','gave'), ('offer','offers','offered'),
                           ('send','sends','sent'), ('tell','tells','told'),
                           ('show','shows','showed')]:
        R('v_ditrans_base', base); R('v_ditrans_sg', sg); R('v_ditrans_past', past)

    return R
