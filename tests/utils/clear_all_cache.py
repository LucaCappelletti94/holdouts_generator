from holdouts_generator import clear_cache, delete_results, clear_work_in_progress

def clear_all_cache(results_directory:str="results"):
    clear_cache()
    clear_work_in_progress(results_directory)
    delete_results(results_directory)