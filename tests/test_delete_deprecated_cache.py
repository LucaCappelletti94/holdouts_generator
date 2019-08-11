from holdouts_generator import random_holdouts, cached_holdouts_generator, clear_cache
from holdouts_generator.utils import delete_deprecated_cache
from glob import glob
import os
import numpy as np

def test_delete_deprecated_cache():
    clear_cache()
    np.random.seed(10)
    generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [1]))
    next(generator())
    os.remove(glob(".holdouts/holdouts/*.pickle.gz")[0])
    delete_deprecated_cache()
    next(generator())
    clear_cache()
    