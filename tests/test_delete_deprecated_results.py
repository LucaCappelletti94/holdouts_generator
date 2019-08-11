from holdouts_generator import store_result, random_holdouts, cached_holdouts_generator, clear_cache, delete_results, delete_deprecated_results
from holdouts_generator.utils import delete_holdout_by_key
from glob import glob
import os
import numpy as np

def make_results():
    generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [3]))
    for i, (_, key, _) in enumerate(generator()):
        store_result(key, {"number":i}, 0)
    return key
        
def make_deprecated_results():
    key = make_results()
    delete_holdout_by_key(key)
    make_results()

def test_delete_deprecated_results():
    delete_results()
    clear_cache()
    make_deprecated_results()
    assert len(delete_deprecated_results()) == 1
    assert len(delete_deprecated_results()) == 0
    delete_results()
    clear_cache()