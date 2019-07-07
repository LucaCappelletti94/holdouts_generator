def pickle_path(cache_directory: str, level:int, number:int)->str:
    return "{cache_directory}/{level}-{number}.pickle".format(
        cache_directory=cache_directory,
        level=level,
        number=number    
    )

def info_path(cache_directory: str)->str:
    return "{cache_directory}/cache.csv".format(
        cache_directory=cache_directory
    )