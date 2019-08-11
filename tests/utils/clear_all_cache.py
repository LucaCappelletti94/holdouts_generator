from holdouts_generator import clear_cache, delete_results, clear_all_work_in_progress

def clear_all_cache(results_directory:str="results"):
    clear_cache()
    clear_all_work_in_progress(results_directory)
    delete_results(results_directory)
