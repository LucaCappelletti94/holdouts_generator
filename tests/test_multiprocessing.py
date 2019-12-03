from holdouts_generator import random_holdouts, cached_holdouts_generator, add_work_in_progress, store_result, skip, regroup_results
import numpy as np
from .utils import clear_all_cache
from multiprocessing import Pool, cpu_count


def job(data, key):
    if data[0] is not None:
        add_work_in_progress("results", key)
        store_result(key, {"ciao": 1}, 0, "results", "holdouts")


def job_wrapper(task):
    job(*task[:2])


def test_multiprocessing():
    clear_all_cache(results_directory="results", cache_dir="holdouts")
    generator = cached_holdouts_generator(
        np.random.randint(100, size=(100, 100)),
        holdouts=random_holdouts([0.1], [24]),
        cache_dir="holdouts",
        skip=skip
    )

    with Pool(cpu_count()) as p:
        p.map(job_wrapper, generator(results_directory="results"))
        p.map(job_wrapper, generator(results_directory="results"))
        p.close()
        p.join()
    regroup_results(results_directory="results")
    clear_all_cache(results_directory="results", cache_dir="holdouts")
