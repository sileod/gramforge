"""Atomic JSON artifacts, with no dependencies on experiment workspaces."""
import json
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def context_text(name_lexicon, symbols):
    text = ('All individuals in this task are people. At least one person exists; '
            'unnamed people may also exist. Definitions quantify over all people, '
            'inside and outside the room.')
    names = [name_lexicon[n] for n in symbols]
    if len(names) == 1:
        text += f' {names[0]} is a person.'
    elif names:
        text += ' ' + ', '.join(names[:-1]) + ' and ' + names[-1] + ' are distinct people.'
    return text
