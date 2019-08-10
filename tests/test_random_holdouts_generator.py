from holdouts_generator import holdouts_generator, clear_cache, cached_holdouts_generator
import numpy as np
from .utils import example_random_holdouts


def run_this_twice():
    x = np.random.randint(10, size=(100))
    generator = holdouts_generator(x, x, holdouts=example_random_holdouts, verbose=False)
    cached_generator = cached_holdouts_generator(x, x, holdouts=example_random_holdouts)
    for ((train, test), inner), ((cached_train, cached_test), _, cached_inner) in zip(generator(), cached_generator()):
        assert [
            np.all(t==ct) for t, ct in zip(train, cached_train)
        ]
        assert [
            np.all(t==ct) for t, ct in zip(test, cached_test)
        ]
        for ((inner_train, inner_test), small), ((inner_cached_train, inner_cached_test), _, cached_small) in zip(inner(), cached_inner()):
            assert [
                np.all(t==ct) for t, ct in zip(inner_train, inner_cached_train)
            ]
            assert [
                np.all(t==ct) for t, ct in zip(inner_test, inner_cached_test)
            ]
            for ((small_train, small_test), _), ((small_cached_train, small_cached_test), _, _) in zip(small(), cached_small()):
                assert [
                    np.all(t==ct) for t, ct in zip(small_train, small_cached_train)
                ]
                assert [
                    np.all(t==ct) for t, ct in zip(small_test, small_cached_test)
                ]


def test_random_holdouts_generator():
    clear_cache()
    run_this_twice()
    run_this_twice()
    clear_cache()
