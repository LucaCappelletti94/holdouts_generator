from holdouts_generator import random_holdouts, chromosomal_holdouts
from random import randint

example_random_holdouts = random_holdouts(
    [0.1, 0.2, 0.1],
    [2, 3, 1]
)

example_chromosomal_holdouts = chromosomal_holdouts([
    (
        [19],
        [([12], None), ([11], None)]
    ),
    (
        [2],
        [([12], [([7], None)]), ([11], [([7], None)])]
    )
])

def skip_all(key:str)->bool:
    return True

def skip_none(key:str)->bool:
    return False