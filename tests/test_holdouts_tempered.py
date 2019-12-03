from holdouts_generator import random_holdouts, cached_holdouts_generator, clear_invalid_cache
from glob import glob
import os
from .utils import clear_all_cache
from touch import touch
import numpy as np
import pytest

def test_holdouts_tempered():
    clear_all_cache(results_directory="results", cache_dir="holdouts")
    np.random.seed(10)
    generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [10]), cache_dir="holdouts")
    list(generator(results_directory="results"))
    paths = glob("holdouts/holdouts/*.pickle.gz")
    path = paths[0]
    os.remove(path)
    touch(path)
    with pytest.raises(ValueError):
        list(generator())
    clear_invalid_cache(cache_dir="holdouts")
    assert set(glob("holdouts/holdouts/*.pickle.gz")) == set(paths[1:])
    list(generator(results_directory="results"))
    paths = glob("holdouts/holdouts/*.pickle.gz")
    path = paths[0]
    os.remove(path)
    with pytest.raises(ValueError):
        list(generator())
    clear_invalid_cache(cache_dir="holdouts")
    assert set(glob("holdouts/holdouts/*.pickle.gz")) == set(paths[1:])
    list(generator(results_directory="results"))
    clear_all_cache(results_directory="results", cache_dir="holdouts")
    