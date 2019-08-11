from holdouts_generator import store_result, random_holdouts, cached_holdouts_generator, delete_all_deprecated_results, skip
from holdouts_generator.utils import delete_holdout_by_key
import numpy as np
import pytest
from .utils import clear_all_cache

def make_results():
    generator = cached_holdouts_generator(np.random.randint(
        100, size=(100, 100)), holdouts=random_holdouts([0.1], [3]), skip=skip)
    hyper_parameters = {
        "test": "testoni"
    }
    for i, ((training, _), key, _) in enumerate(generator(hyper_parameters)):
        if training is not None:
            store_result(key, {"number": i}, 0, hyper_parameters=hyper_parameters)
            with pytest.raises(ValueError):
                store_result(key, {"number": i}, 0, hyper_parameters=hyper_parameters)
    return key


def make_deprecated_results():
    key = make_results()
    delete_holdout_by_key(key)
    make_results()


def test_delete_deprecated_results():
    clear_all_cache()
    make_deprecated_results()
    assert len(delete_all_deprecated_results(".holdouts", "results")) == 1
    assert len(delete_all_deprecated_results(".holdouts", "results")) == 0
    clear_all_cache()