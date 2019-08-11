from holdouts_generator import merge_all_results, cached_holdouts_generator, random_holdouts, store_result, delete_all_duplicate_results
from typing import Callable
import numpy as np
from .utils import clear_all_cache

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
	directory_A = "root/results_A"
	directory_B = "root/results_B"
	target = "results_target"
	clear_all_cache(directory_A)
	clear_all_cache(directory_B)
	clear_all_cache(target)
	generator = cached_holdouts_generator(np.random.randint(100, size=(100,100)), holdouts=random_holdouts([0.1], [3]))
	make_results(generator, directory_A)
	make_results(generator, directory_B)
	merge_all_results("root", target)
	delete_all_duplicate_results("root")
	clear_all_cache(directory_A)
	clear_all_cache(directory_B)
	clear_all_cache(target)
