from holdouts_generator import random_holdouts, chromosomal_holdouts, balanced_random_holdouts
from random import randint
from .clear_all_cache import clear_all_cache
from .slow_random_holdout import slow_random_holdouts

example_random_holdouts = random_holdouts(
    [0.1, 0.2, 0.3],
    [2, 1, 1],
    hyper_parameters={"type":"random"}
)

example_random_holdouts_2 = random_holdouts(
    [0.1, 0.2, 0.3],
    [2, 1, 1],
    hyper_parameters={"type":"random2"}
)

example_balanced_random_holdouts = balanced_random_holdouts(
    [0.1, 0.2, 0.3],
    [2, 1, 1],
    hyper_parameters={"type":"balanced"}
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
], hyper_parameters={"type":"chromosomal_holdouts"})