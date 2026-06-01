"""
Constraint:
all elements should be logically independant
for all p, q in the predicates/propositions
p (atom or predicates) does not entail nor contradict q or ~q
even if correlated or linked
"""

def negate_predicate(predicate):
    replacements = {
        "is ": "is not ",
        "has ": "does not have ",
        "does ": "does not ",
        'can ': 'cannot '
    }
    for key, val in replacements.items():
        if predicate.startswith(key):
            return predicate.replace(key, val, 1)

    if predicate.endswith((" marked", " tagged")):
        return "is not " + predicate
    
    words = predicate.split(" ")
    if words[0].endswith("s"):
        words[0] = words[0][:-1]  # Remove 's' from the verb
    return "does not " + " ".join(words)



short_propositions = [
    "Planet Xylos has diamond rain.",
    "Bellbridge's houses are all purple.",
    "Gravity inverts in Oakhaven on Tuesdays.",
    "A tree in Whispering Woods has golden fruit.",
    "A square cloud is over Silver Lake.",
    "The lighthouse on Cape Sorrow glows green.",
    "The Great Library of Alexandria still exists.",
    "A singing flower blooms in the Amazon.",
    "John Smith's car runs on ethanol.",
    "The clock tower in Chronos strikes thirteen times."
]

# neg of short_propositions, same order
neg_short_propositions = [
    "Planet Xylos has no diamond rain.",
    "Not all Bellbridge's houses are purple.",
    "Gravity does not invert in Oakhaven on Tuesdays.",
    "No tree in Whispering Woods has golden fruit.",
    "No square cloud is over Silver Lake.",
    "The lighthouse on Cape Sorrow does not glow green.",
    "The Great Library of Alexandria does not exist.",
    "No singing flower blooms in the Amazon.",
    "John Smith's car does not run on ethanol.",
    "The clock tower in Chronos does not strike thirteen times."
]

predicates = [
    "alpha tagged", "bravo tagged", "charlie tagged", "delta tagged", "echo tagged",
    "foxtrot tagged", "golf tagged", "hotel tagged", "india tagged", "juliet tagged",
    "kilo tagged", "lima tagged", "mike tagged", "november tagged", "oscar tagged",
    "papa tagged", "quebec tagged", "romeo tagged", "sierra tagged", "tango tagged",
    "uniform tagged", "victor tagged", "whiskey tagged", "xray tagged", "yankee tagged",
    "zulu tagged",
]
