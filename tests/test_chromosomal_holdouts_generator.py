from holdouts_generator import holdouts_generator, cached_holdouts_generator
import pandas as pd
import numpy as np
from .utils import example_chromosomal_holdouts, clear_all_cache

def run_this_twice():
	x = pd.read_csv("test_dataset/x.csv", index_col=0)
	generator = holdouts_generator(x, x, holdouts=example_chromosomal_holdouts)
	cached_generator = cached_holdouts_generator(x, x, holdouts=example_chromosomal_holdouts, cache_dir="holdouts")
	for ((train, test), inner), ((cached_train, cached_test), _, cached_inner) in zip(generator(), cached_generator()):
		assert all([
			np.all(t==ct) for t, ct in zip(train, cached_train)
		])
		assert all([
			np.all(t==ct) for t, ct in zip(test, cached_test)
		])
		for ((inner_train, inner_test), _), ((inner_cached_train, inner_cached_test), _, _) in zip(inner(), cached_inner()):
			assert all([
				np.all(t==ct) for t, ct in zip(inner_train, inner_cached_train)
			])
			assert all([
				np.all(t==ct) for t, ct in zip(inner_test, inner_cached_test)
			])

def test_chromosomal_holdouts_generator():
	clear_all_cache(results_directory="results", cache_dir="holdouts")
	run_this_twice()
	run_this_twice()
	clear_all_cache(results_directory="results", cache_dir="holdouts")