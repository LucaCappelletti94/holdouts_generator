import os
import pickle
import shutil
from sklearn.model_selection import train_test_split
from auto_tqdm import tqdm
from typing import List, Tuple

def _get_holdout(dataset, test_size:float, random_state:int, cache:bool, cache_dir:str):
    """Return given holdout, also handles cache if it is enabled."""
    if cache:
        path = "{cache_dir}/holdout_{test_size}_{random_state}.pickle".format(
            cache_dir=cache_dir,
            test_size=test_size,
            random_state=random_state
        )
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            os.makedirs(cache_dir, exist_ok=True)
            data = train_test_split(*dataset, test_size=test_size, random_state=random_state)
            with open(path, "wb") as f:
                pickle.dump(data, f)
            return data
    return train_test_split(*dataset, test_size=test_size, random_state=random_state)        

def holdouts_generator(*dataset, test_sizes:List[float]=None, holdouts:List[int]=None, random_state:int=42, verbose:bool=True, cache:bool=False, cache_dir:str=".holdouts_cache"):
    """Return validation dataset and another holdout generator
        dataset, iterable of datasets to generate holdouts from.
        test_sizes:List[float]=None, list of floats from 0 to 1, representing how many datapoints should be reserved to the test set.
        holdouts:List[int]=None, list of holdouts sizes.
        random_state:int=42, random state to reproduce experiment.
        verbose:bool=True, whetever to show or not loading bars.
        cache:bool=False, whetever to cache or not the rendered holdouts.
        cache_dir:str=".cache", directory where to cache the holdouts.
    """
    assert len(test_sizes) > 0
    assert len(test_sizes) == len(holdouts)
    test_size = test_sizes.pop(0)
    holdouts_number = holdouts.pop(0)

    def generator():
        for i in tqdm(range(holdouts_number), verbose=verbose):
            validation = _get_holdout(dataset, test_size, random_state+i, cache, cache_dir)
            if not holdouts:
                yield validation
            else:
                yield validation, holdouts_generator(
                    *[v for i, v in enumerate(validation) if i%2==0],
                    test_sizes=test_sizes,
                    holdouts=holdouts,
                    random_state=random_state,
                    verbose=verbose,
                    cache=cache,
                    cache_dir="{cache_dir}/{i}".format(
                        cache_dir=cache_dir,
                        i=i
                    )
                )
    return generator


def clear_holdouts_cache(cache_dir:str=".holdouts_cache"):
    """Remove the holdouts cace directory.
        cache_dir:str=".holdouts_cache", the holdouts cache directory to be removed.
    """
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)