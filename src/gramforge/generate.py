from .generate_recursive import generate_recursive
from .generate_sequential import generate_sequential
import random

def generate(start, depth=10, n_iter=10_000, mode='sequential', seed=None, min_depth=None, *args, **kwargs):
    random.seed(seed)
    if type(start)==type:
        start=start.start()

    modes = {'recursive': generate_recursive,
              'sequential': generate_sequential,
            }

    generate = modes[mode]    
    """Generate one production using specified mode."""
    for _ in range(n_iter):
        result = generate(start, max_depth=depth, min_depth=min_depth, *args, **kwargs)
        if min_depth and result and result[0].height < min_depth:
            continue
        if result: return result[0]
    raise ValueError('Incomplete generation')