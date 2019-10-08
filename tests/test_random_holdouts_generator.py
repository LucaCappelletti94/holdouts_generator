from holdouts_generator import holdouts_generator, cached_holdouts_generator
import numpy as np
from .utils import example_random_holdouts, clear_all_cache


def run_this_twice():
    x = np.random.RandomState(seed=42).randint(10, size=(100))
    generator = holdouts_generator(x, x, holdouts=example_random_holdouts, verbose=False)
    cached_generator = cached_holdouts_generator(x, x, holdouts=example_random_holdouts, cache_dir="holdouts")
    for ((train, test), inner), ((cached_train, cached_test), _, cached_inner) in zip(generator(), cached_generator()):
        assert all([
            np.all(t==ct) for t, ct in zip(train, cached_train)
        ])
        assert all([
            np.all(t==ct) for t, ct in zip(test, cached_test)
        ])
        for ((inner_train, inner_test), small), ((inner_cached_train, inner_cached_test), _, cached_small) in zip(inner(), cached_inner()):
            assert all([
                np.all(t==ct) for t, ct in zip(inner_train, inner_cached_train)
            ])
            assert all([
                np.all(t==ct) for t, ct in zip(inner_test, inner_cached_test)
            ])
            for ((small_train, small_test), _), ((small_cached_train, small_cached_test), _, _) in zip(small(), cached_small()):
                assert all([
                    np.all(t==ct) for t, ct in zip(small_train, small_cached_train)
                ])
                assert all([
                    np.all(t==ct) for t, ct in zip(small_test, small_cached_test)
                ])


def test_random_holdouts_generator():
    clear_all_cache(results_directory="results", cache_dir="holdouts")
    run_this_twice()
    run_this_twice()
    clear_all_cache(results_directory="results", cache_dir="holdouts")