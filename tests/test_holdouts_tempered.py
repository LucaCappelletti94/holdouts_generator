from holdouts_generator import store_result, random_holdouts, cached_holdouts_generator, clear_cache
from glob import glob
import os
import numpy as np
import pytest

def test_holdouts_tempered():
    clear_cache()
    np.random.seed(10)
    generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [1]))
    next(generator())
    next(generator())
    os.remove(glob(".holdouts/holdouts/*.pickle.gz")[0])
    with pytest.raises(ValueError):
        next(generator())
    clear_cache()
    