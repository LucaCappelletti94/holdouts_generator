from holdouts_generator import random_holdouts, cached_holdouts_generator, add_work_in_progress, store_result
import numpy as np
from .utils import clear_all_cache
from multiprocessing import Pool, cpu_count

def job(key):
    add_work_in_progress(key)
    store_result(key, {"ciao": 1}, 0)

def job_wrapper(task):
    job(task[1])

def test_work_in_progress():
    clear_all_cache()
    holdouts = list(cached_holdouts_generator(np.random.randint(
        100, size=(100, 100)), holdouts=random_holdouts([0.1], [12]))())
    
    with Pool(cpu_count()) as p:
        p.map(job_wrapper, holdouts)

    clear_all_cache()