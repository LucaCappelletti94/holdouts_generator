from holdouts_generator import random_holdouts, cached_holdouts_generator, add_work_in_progress, store_result, remove_work_in_progress
import numpy as np
import pytest
from .utils import clear_all_cache


def test_work_in_progress():
    clear_all_cache(results_directory="results", cache_dir="holdouts")
    generator = cached_holdouts_generator(np.random.randint(
        100, size=(100, 100)), holdouts=random_holdouts([0.1], [3]), cache_dir="holdouts")
    for _, key, _ in generator(results_directory="results"):
        with pytest.raises(ValueError):
            remove_work_in_progress("results", key)
        add_work_in_progress("results", key)
        store_result(key, {"ciao": 1}, 0, results_directory="results", cache_dir="holdouts")
        with pytest.raises(ValueError):
            add_work_in_progress("results", key)
    clear_all_cache(results_directory="results", cache_dir="holdouts")
