from holdouts_generator import random_holdouts, cached_holdouts_generator, clear_cache, delete_results, add_work_in_progress, clear_work_in_progress
import numpy as np
import pytest

def test_work_in_progress():
    clear_cache()
    delete_results()
    clear_work_in_progress()
    generator = cached_holdouts_generator(np.random.randint(
        100, size=(100, 100)), holdouts=random_holdouts([0.1], [3]))
    for _, key, _ in generator():
        add_work_in_progress(key)
        with pytest.raises(ValueError):
            add_work_in_progress(key)
    clear_cache()
    clear_work_in_progress()
    delete_results()