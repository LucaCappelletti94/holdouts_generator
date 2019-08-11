from .utils import work_in_progress_path, build_query, build_keys
import pandas as pd
import os
from typing import Dict


def load_work_in_progress(results_directory: str = "results"):
    """Load work in progress holdouts for given results directory.
        results_directory: str = "results", directory where results are stored.
    """
    return pd.read_csv(work_in_progress_path(results_directory))


def store_work_in_progress(wip: pd.DataFrame, results_directory: str):
    """Store work in progress holdouts for given results directory.
        wip: pd.DataFrame, standard dataframe of work in progress to store.
        results_directory: str = "results", directory where results are stored.
    """
    os.makedirs(results_directory, exist_ok=True)
    wip.to_csv(work_in_progress_path(results_directory), index=False)


def add_work_in_progress(key: str, hyper_parameters: Dict = None, results_directory: str = "results"):
    """Sign given holdout key as under processing for given results directory.
        key: str, key identifier of holdout.
        hyper_parameters: Dict, hyper parameters to check for.
        results_directory: str = "results", directory where results are stored.
    """
    if is_work_in_progress(key, hyper_parameters, results_directory):
        raise ValueError("Given key {key} for given directory {results_directory} is already work in progress!".format(
            key=key,
            results_directory=results_directory
        ))
    new_row = pd.DataFrame(
        build_keys(key, hyper_parameters),
        index=[0]
    )
    try:
        wip = pd.concat([
            load_work_in_progress(results_directory),
            new_row
        ])
    except FileNotFoundError:
        wip = new_row
    store_work_in_progress(wip, results_directory)


def is_work_in_progress(key: str, hyper_parameters:Dict=None, results_directory: str = "results") -> bool:
    """Return boolean representing if given key is under work for given results directory.
        key: str, key identifier of holdout.
        results_directory: str = "results", directory where results are stored.
    """
    return os.path.isfile(work_in_progress_path(results_directory)) and not load_work_in_progress(results_directory).query(build_query(build_keys(key, hyper_parameters))).empty


def clear_work_in_progress(results_directory: str = "results"):
    """Delete work in progress log for given results directory.
        results_directory: str = "results", directory where results are stored.
    """
    if os.path.exists(work_in_progress_path(results_directory)):
        os.remove(work_in_progress_path(results_directory))
