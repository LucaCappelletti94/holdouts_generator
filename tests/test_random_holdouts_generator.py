from holdouts_generator import holdouts_generator, clear_cache, random_holdouts
import numpy as np


def test_test_random_holdouts_generator():
    dataset = np.random.randint(10, size=(2, 100, 10))
    generator = holdouts_generator(
        *dataset,
        holdouts=random_holdouts(
            [0.1, 0.2, 0.1],
            [2, 3, 1]
        )
    )
    for _, s1 in generator():
        for _, s2 in s1():
            for _ in s2():
                pass
    list(generator())
    generator = holdouts_generator(
        *dataset,
        holdouts=random_holdouts(
            [0.1, 0.2],
            [2, 3]
        ),
        cache=True
    )
    list(generator())
    list(generator())
    clear_cache()