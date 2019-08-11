from holdouts_generator import merge_results, cached_holdouts_generator, random_holdouts, store_result, delete_results, load_results
from typing import Callable
import numpy as np

def make_results(generator:Callable, results_directory:str):
    for i, (_, key, _) in enumerate(generator(results_directory=results_directory)):
        store_result(key, {
			"number":i,
			"directory": results_directory
		}, 0, {
            "test": results_directory
        }, results_directory=results_directory)

def test_merge_results():
	np.random.seed(10)
	directory_A = "results_A"
	directory_B = "results_B"
	target = "results_target"
	generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [3]))
	make_results(generator, directory_A)
	make_results(generator, directory_B)
	merge_results([directory_A, directory_B], target)
	assert load_results(directory_A).size + load_results(directory_B).size == load_results(target).size
	delete_results(directory_A)
	delete_results(directory_B)
	delete_results(target)
