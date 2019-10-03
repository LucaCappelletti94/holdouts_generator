from holdouts_generator import random_holdouts, chromosomal_holdouts, balanced_random_holdouts
from random import randint
from .clear_all_cache import clear_all_cache

example_random_holdouts = random_holdouts(
    [0.1, 0.2, 0.3],
    [2, 1, 1]
)

example_balanced_random_holdouts = balanced_random_holdouts(
    [0.1, 0.2, 0.3],
    [2, 1, 1]
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