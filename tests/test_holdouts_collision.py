from holdouts_generator import random_holdouts, cached_holdouts_generator, add_work_in_progress, store_result, skip, regroup_results
import numpy as np
from .utils import clear_all_cache, slow_random_holdouts
from multiprocessing import Pool, cpu_count
from time import sleep
import random


def job(X):
    sleep(random.random())
    list(cached_holdouts_generator(X, skip=skip, holdouts=slow_random_holdouts([0.1, 0.1], [cpu_count(), 1]))())


def test_work_in_progress():
    clear_all_cache()
    X = np.random.randint(100, size=(100, 100))

    with Pool(cpu_count()) as p:
        p.map(job, [X]*cpu_count())
        p.close()
        p.join()
    regroup_results()
    clear_all_cache()
