from holdouts_generator import random_holdouts, cached_holdouts_generator, add_work_in_progress
import numpy as np
import pytest
from .utils import clear_all_cache

def test_work_in_progress():
    clear_all_cache()
    generator = cached_holdouts_generator(np.random.randint(
        100, size=(100, 100)), holdouts=random_holdouts([0.1], [3]))
    for _, key, _ in generator():
        add_work_in_progress(key)
        with pytest.raises(ValueError):
            add_work_in_progress(key)
    clear_all_cache()