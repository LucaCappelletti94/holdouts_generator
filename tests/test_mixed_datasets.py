from holdouts_generator import cached_holdouts_generator
import pandas as pd
import numpy as np
from glob import glob
from .utils import example_chromosomal_holdouts, clear_all_cache, example_random_holdouts, example_balanced_random_holdouts, example_random_holdouts_2

def run_this_twice():
    chromo = pd.read_csv("test_dataset/x.csv", index_col=0)
    x = np.random.RandomState(seed=42).randint(10, size=(100))
    list(cached_holdouts_generator(chromo, holdouts=example_chromosomal_holdouts, cache_dir="holdouts")())
    list(cached_holdouts_generator(x, holdouts=example_random_holdouts, cache_dir="holdouts")())
    list(cached_holdouts_generator(x, holdouts=example_random_holdouts_2, cache_dir="holdouts")())
    list(cached_holdouts_generator(x, holdouts=example_balanced_random_holdouts, cache_dir="holdouts")())


def test_chromosomal_holdouts_generator():
    clear_all_cache(results_directory="results", cache_dir="holdouts")
    run_this_twice()
    run_this_twice()
    assert 8 == len([
        path for path in glob("holdouts/cache/*.json")
    ])
    clear_all_cache(results_directory="results", cache_dir="holdouts")