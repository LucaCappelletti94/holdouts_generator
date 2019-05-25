from holdouts_generator import holdouts_generator, clear_holdouts_cache, random_holdouts
import numpy as np

def test_test_random_holdouts_generator():
    dataset = np.random.randint(10, size=(100,10))
    generator = holdouts_generator(
        *dataset, 
        holdouts=random_holdouts(
            [0.1, 0.2],
            [2, 3]
        )
    )
    list(generator())
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
    clear_holdouts_cache()