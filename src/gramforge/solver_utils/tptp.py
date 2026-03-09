import os, io, subprocess, random, tempfile
from functools import lru_cache
from easydict import EasyDict as edict
import stat, shutil
from appdirs import user_data_dir
import pooch
import json

def _vampire_works(path):
    """Return True if the binary at *path* actually runs."""
    try:
        subprocess.run([path, "--version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


def _build_vampire(cache_dir):
    """Clone & build Vampire from source; return path to the binary."""
    src = os.path.join(cache_dir, "vampire-src")
    build = os.path.join(src, "build")
    binary = os.path.join(build, "bin", "vampire")
    if os.path.isfile(binary) and _vampire_works(binary):
        return binary
    shutil.rmtree(src, ignore_errors=True)
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/vprover/vampire.git", src])
    os.makedirs(build, exist_ok=True)
    subprocess.check_call(["cmake", ".."], cwd=build)
    subprocess.check_call(["make", f"-j{os.cpu_count() or 1}"], cwd=build)
    if not os.path.isfile(binary):
        # some cmake configs put it directly in build/
        alt = os.path.join(build, "vampire")
        if os.path.isfile(alt):
            binary = alt
    assert os.path.isfile(binary), f"Vampire build failed – no binary at {binary}"
    return binary


def get_vampire_path():
    # 1. already on PATH
    path = shutil.which("vampire")
    if path and _vampire_works(path):
        return path

    cache_dir = user_data_dir("vampire-wrapper")
    os.makedirs(cache_dir, exist_ok=True)

    # 2. try prebuilt release download
    try:
        vampire_path = pooch.retrieve(
            url="https://github.com/vprover/vampire/releases/download/v4.9casc2024/vampire",
            fname="vampire",
            path=cache_dir,
            known_hash=None,
        )
        os.chmod(vampire_path, os.stat(vampire_path).st_mode | stat.S_IEXEC)
        if _vampire_works(vampire_path):
            return vampire_path
    except Exception:
        pass

    # 3. fallback: build from source
    return _build_vampire(cache_dir)


VAMPIRE_PATH = get_vampire_path()

class Serializable(dict):
    """
    Drop-in mix-in: any subclass becomes automatically json.dumps-able.
    """

    def __setattr__(self, key, value):
        super().__setattr__(key, value)   # keep normal attribute behaviour
        self[key] = value                 # keep the dict in sync

    def __delattr__(self, key):
        super().__delattr__(key)
        self.pop(key, None)

    # nice-to-have helper
    def to_json(self, **dump_kw):
        return json.dumps(self, **dump_kw)

class ProofOutput(Serializable):
    def __init__(self,proof, input=''):
        self.proof = proof
        self.rules, self.indices = extract_inferences_and_formulas(proof)
        self.status = proof.split(' SZS status ',1)[-1].split(' for ')[0]
        self.sat = self.status == "Satisfiable"
        self.input = input

        if "Time limit" in self.status:
            self.status = "Time limit"
        if "Refutation not found" in self.status:
            self.status = "Refutation not found"

    def __str__(self):
        return f"{self.status}" + (f":{len(self.indices)}" if self.indices else "")
    __repr__=__str__

    def to_dict(self):
        return edict({k: getattr(self,k) for k in self.__dict__  if not k.startswith('__')})

    def to_json(self):
        return json.dumps(self.to_dict())


def split_clauses(x,prefix='axiom',name_prefix='',do_split=True):
    clauses=x.split('&\n')
    if any(a.count('(')!=a.count(')') for a in clauses):
        clauses=[x]
    return '\n'.join([f"fof({name_prefix}{i},{prefix},{c})." for i,c in enumerate(clauses)])+"\n"


def to_tptp(x,background='',problem='prem',neg='',mode='sat',use_hypothesis=True):
    mode={'sat':'axiom','proof':'conjecture'}[mode]
    premise = split_clauses(x.tptp,prefix=mode,name_prefix="p")
    if use_hypothesis:
        hypothesis = f"fof(hypothesis,{mode},{neg}({x.hyp_tptp}))."
    else:
        hypothesis=""
    return f"{background}\n{premise}\n{hypothesis}".replace('¿','?')

def run(expr, solver='vampire', proof=False, verbose=True):
    if not expr.strip().endswith(').') and not expr.strip().startswith('fof'):
        expr = f"fof(expr,axiom,{expr})."
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, dir="/dev/shm/") as f:
        f.write(expr)
        path = f.name
    
    try:
        cmd = [VAMPIRE_PATH, path, "--output_axiom_names", "on", "-t", "20s"]
        result = subprocess.run(cmd, text=True, capture_output=True)
        output = result.stdout if verbose else result.stdout.split(' SZS status ', 1)[-1].split(' for ')[0]
        return ProofOutput(output, input=expr)
    finally:
        os.remove(path)




def extract_inferences_and_formulas(proof):
    inferences,inputs=[],[]
    for x in proof.split('\n'):
        if not x.endswith(']'): 
            continue
        x=x.split('[')[-1].strip(']')        
        if x.startswith('input'):
            inputs.append(x.replace('input ',''))
        inferences.append(x.rsplit(' ',-1)[0])
    return inferences,inputs

def extract_inferences_and_formulas_tff(tff_statements):
    inference_types = list()
    formula_names = list()
    for line in tff_statements.strip().split("\n"):
        if "inference" in line:
            inference_types.append(line.split("inference(")[1].split(",")[0])
        if "file" in line:
            formula_names.append(line.split("file")[1].split(",")[1].split("'")[1].split('/')[-1])
    return inference_types, formula_names