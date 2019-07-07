from holdouts_generator import holdouts_generator, clear_cache, chromosomal_holdouts, cached_holdouts_generator
import pandas as pd
from .utils import example_chromosomal_holdouts


def test_chromosomal_holdouts_generator():
    x = pd.read_csv("test_dataset/x.csv", index_col=0)
    generator = holdouts_generator(x, x, holdouts=example_chromosomal_holdouts)
    for _, s1 in generator():
        for _ in s1():
            pass
    generator = cached_holdouts_generator(x, x, holdouts=example_chromosomal_holdouts)
    list(generator())
    list(generator())
    clear_cache()