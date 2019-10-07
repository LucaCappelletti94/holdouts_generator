from holdouts_generator import random_holdouts, cached_holdouts_generator, clear_invalid_cache
from glob import glob
import os
from .utils import clear_all_cache
from touch import touch
import numpy as np
import pytest

def test_holdouts_tempered():
    clear_all_cache()
    np.random.seed(10)
    generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [1]))
    next(generator())
    next(generator())
    path = glob(".holdouts/holdouts/*.pickle.gz")[0]
    os.remove(path)
    touch(path)
    with pytest.raises(ValueError):
        next(generator())
    clear_invalid_cache()
    next(generator())
    path = glob(".holdouts/holdouts/*.pickle.gz")[0]
    os.remove(path)
    with pytest.raises(ValueError):
        next(generator())
    clear_invalid_cache()
    next(generator())
    clear_all_cache()
    