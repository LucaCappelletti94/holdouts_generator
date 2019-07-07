from typing import Callable, List
from .paths import pickle_path, info_path
from .various import odd_even_split
from .hash import hash_file
import os
import pandas as pd
import pickle

def uncached(generator:Callable, dataset:List, *args):
    return odd_even_split(generator(dataset))

def cached(generator:Callable, dataset:List, cache_dir:str, level:int, number:int):
    try:
        return load(pickle_path(cache_dir, level, number))
    except (pickle.PickleError, FileNotFoundError):
        data = odd_even_split(generator(dataset))
    key = dump(data, cache_dir, level, number)
    return (*data, key)

def load(path:str):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_info(path:str, level:int, number:int, key:str)->pd.DataFrame:
    return pd.DataFrame({
        "path":path,
        "level":level,
        "number":number,
        "key":key
    }, index=[0])

def dump(data, cache_dir:str, level:int, number:int)->str:
    path = pickle_path(cache_dir, level, number)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    key = hash_file(path)
    info_file = info_path(cache_dir)
    info = build_info(path, level, number, key)
    if os.path.exists(info_file):
        info = pd.concat([pd.read_csv(info_file), info])
    info.to_csv(info_file, index=False)
    return key

def get_holdout_key(cache_dir:str, level:int, number:int)->str:
    """Return key, if cached, for given holdout.
        cache_dir:str, cache directory to load data from
        level:int, level of given holdout.
        number:int, number of given holdout.
    """
    try:
        return pd.read_csv(info_path(cache_dir)).query(
            'level == {level} & number == {number}'.format(
                level=level,
                number=number
            )
        )["key"].values[0]
    except (FileNotFoundError, IndexError):
        pass