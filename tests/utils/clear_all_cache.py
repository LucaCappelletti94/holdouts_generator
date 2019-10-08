from holdouts_generator import clear_cache, delete_results, clear_work_in_progress

def clear_all_cache(results_directory:str="results", cache_dir:str="holdouts"):
    """
        results_directory: str, directory where to store the results.
        cache_dir:str, the holdouts cache directory.
    """
    clear_cache(cache_dir)
    clear_work_in_progress(results_directory)
    delete_results(results_directory)
