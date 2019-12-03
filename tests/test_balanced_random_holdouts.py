from holdouts_generator import holdouts_generator, cached_holdouts_generator, balanced_random_holdouts
import numpy as np
from .utils import example_balanced_random_holdouts, clear_all_cache


def run_this_twice():
    x = np.random.RandomState(seed=42).randint(10, size=(100))
    generator = holdouts_generator(
        x, x, holdouts=example_balanced_random_holdouts, verbose=False)
    cached_generator = cached_holdouts_generator(
        x, x, holdouts=example_balanced_random_holdouts, cache_dir="holdouts")
    for ((train, test), inner), ((cached_train, cached_test), _, cached_inner) in zip(generator(), cached_generator(results_directory="results")):
        assert all([
            t.shape == ct.shape and np.all(t == ct) for t, ct in zip(train, cached_train)
        ])
        assert all([
            t.shape == ct.shape and np.all(t == ct) for t, ct in zip(test, cached_test)
        ])
        for ((inner_train, inner_test), small), ((inner_cached_train, inner_cached_test), _, cached_small) in zip(inner(), cached_inner(results_directory="results")):
            assert all([
                t.shape == ct.shape and np.all(t == ct) for t, ct in zip(inner_train, inner_cached_train)
            ])
            assert all([
                t.shape == ct.shape and np.all(t == ct) for t, ct in zip(inner_test, inner_cached_test)
            ])
            for ((small_train, small_test), _), ((small_cached_train, small_cached_test), _, _) in zip(small(), cached_small(results_directory="results")):
                assert all([
                    t.shape == ct.shape and np.all(t == ct) for t, ct in zip(small_train, small_cached_train)
                ])
                assert all([
                    t.shape == ct.shape and np.all(t == ct) for t, ct in zip(small_test, small_cached_test)
                ])


def test_balanced_random_holdouts_generator():
    clear_all_cache(results_directory="results", cache_dir="holdouts")
    run_this_twice()
    run_this_twice()
    clear_all_cache(results_directory="results", cache_dir="holdouts")


def test_balanced_random_holdouts_generator_alignment():
    y = np.hstack([
        np.ones(100),
        np.zeros(100)
    ])
    x = np.arange(200)
    ((train_x, train_y), (test_x, test_y)), _ = next(
        holdouts_generator(
            x, y,
            holdouts=balanced_random_holdouts([0.3], [1])
        )(results_directory="results"))
    assert (y[train_x] == train_y).all()
    assert (y[test_x] == test_y).all()
