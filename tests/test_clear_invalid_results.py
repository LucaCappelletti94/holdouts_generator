from holdouts_generator import random_holdouts, cached_holdouts_generator, store_result, clear_invalid_results
from glob import glob
import os
from .utils import clear_all_cache
import numpy as np
import pytest

def test_clear_invalid_results():
    clear_all_cache()
    np.random.seed(10)
    generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [1]))
    gen = generator()
    (_, _), key, _ = next(gen)
    store_result(key, {"ping":"pong"}, 0, hyper_parameters={"keb":"ab"})
    assert len(glob("results/results/*.json")) == 1
    clear_invalid_results()
    assert len(glob("results/results/*.json")) == 1
    path = glob(".holdouts/holdouts/*.pickle.gz")[0]
    os.remove(path)
    with pytest.raises(ValueError):
        list(generator())
    clear_invalid_results()
    with pytest.raises(ValueError):
        store_result(key, {"ping":"pong"}, 0)
    assert len(glob("results/results/*.json")) == 0
    clear_all_cache()
    