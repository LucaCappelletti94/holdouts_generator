from holdouts_generator import random_holdouts, chromosomal_holdouts

example_random_holdouts = random_holdouts(
    [0.1, 0.2, 0.1],
    [2, 3, 1]
)

example_chromosomal_holdouts = chromosomal_holdouts([
    (
        [19],
        [([12], [([7], None)]), ([11], [([7], None)])]
    ),
    (
        [2],
        [([12], [([7], None)]), ([11], [([7], None)])]
    )
])

def skip(key:str)->bool:
    return True