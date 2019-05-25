from holdouts_generator import holdouts_generator, clear_holdouts_cache
import numpy as np

def test_holdouts_generator():
    dataset = np.random.randint(10, size=(100,10))
    generator = holdouts_generator(*dataset, test_sizes=[0.3], holdouts=[3])
    list(generator())
    list(generator())
    generator = holdouts_generator(*dataset, test_sizes=[0.3], holdouts=[3], cache=True)
    list(generator())
    list(generator())
    clear_holdouts_cache()