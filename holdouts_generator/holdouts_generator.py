import os
import pickle
import shutil
from auto_tqdm import tqdm
from typing import List

def get_desc(level:int):
    if level==0:
        return "Holdouts"
    if level==1:
        return "Inner holdouts"
    if level>1:
        return "Inner holdouts (level {level})".format(level=level)


def holdouts_generator(*dataset:List, holdouts:List, verbose:bool=True, cache:bool=False, cache_dir:str=".holdouts", level:int=0):
    """Return validation dataset and another holdout generator
        dataset, iterable of datasets to generate holdouts from.
        holdouts:List, list of holdouts callbacks.
        verbose:bool=True, whetever to show or not loading bars.
        cache:bool=False, whetever to cache or not the rendered holdouts.
        cache_dir:str=".cache", directory where to cache the holdouts.
    """
    if holdouts is None:
        return None
    def generator():
        for outer_holdout, name, inner_holdouts in tqdm(list(holdouts), verbose=verbose, desc=get_desc(level)):
            path = "{cache_dir}/{name}".format(cache_dir=cache_dir, name=name)
            pickle_path = "{path}.pickle".format(path=path)
            if cache and os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    validation = pickle.load(f)
            validation = outer_holdout(dataset)
            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                with open(pickle_path, "wb") as f:
                    pickle.dump(validation, f)

            training, testing = validation[::2], validation[1::2]
            yield (training, testing), holdouts_generator(
                *training,
                holdouts=inner_holdouts,
                verbose=verbose,
                cache=cache,
                cache_dir=path,
                level=level+1
            )
    return generator


def clear_holdouts_cache(cache_dir:str=".holdouts"):
    """Remove the holdouts cace directory.
        cache_dir:str=".holdouts", the holdouts cache directory to be removed.
    """
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)