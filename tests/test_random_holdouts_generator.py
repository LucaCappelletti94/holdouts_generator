from holdouts_generator import holdouts_generator, clear_cache, cached_holdouts_generator
import numpy as np
from .utils import example_random_holdouts, skip_all, skip_none

def test_test_random_holdouts_generator():
    dataset = np.random.randint(10, size=(2, 100, 10))
    generator = holdouts_generator(*dataset, holdouts=example_random_holdouts)
    for _, _, s1 in generator():
        for _ in s1():
            pass
    list(generator())
    generator = cached_holdouts_generator(*dataset, holdouts=example_random_holdouts, skip=skip_all)
    list(generator())
    list(generator())
    generator = cached_holdouts_generator(*dataset, holdouts=example_random_holdouts, skip=skip_none)
    list(generator())
    list(generator())
    clear_cache()