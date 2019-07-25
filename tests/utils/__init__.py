from holdouts_generator import random_holdouts, chromosomal_holdouts
from random import randint

example_random_holdouts = random_holdouts(
    [0.1, 0.2],
    [2, 1]
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

def skip_all(*args, **kwargs)->bool:
    return True

def skip_none(*args, **kwargs)->bool:
    return False