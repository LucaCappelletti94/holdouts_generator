from holdouts_generator import random_holdouts, cached_holdouts_generator
from glob import glob
import os
from .utils import clear_all_cache
import numpy as np
import pytest

def test_holdouts_tempered():
    clear_all_cache()
    np.random.seed(10)
    generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [1]))
    next(generator())
    next(generator())
    os.remove(glob(".holdouts/holdouts/*.pickle.gz")[0])
    with pytest.raises(ValueError):
        next(generator())
    clear_all_cache()
    