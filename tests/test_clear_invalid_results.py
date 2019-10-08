from holdouts_generator import random_holdouts, cached_holdouts_generator, store_result, clear_invalid_results
from glob import glob
import os
from .utils import clear_all_cache
import numpy as np
import pytest

def test_holdouts_tempered():
    clear_all_cache()
    np.random.seed(10)
    generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [2]))
    gen = generator()
    (_, _), key, _ = next(gen)
    store_result(key, {"ping":"pong"}, 0)
    clear_invalid_results()
    _ = next(gen)
    path = glob(".holdouts/holdouts/*.pickle.gz")[0]
    os.remove(path)
    with pytest.raises(ValueError):
        next(generator())
    clear_invalid_results()
    clear_all_cache()
    