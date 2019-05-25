from holdouts_generator import holdouts_generator, clear_holdouts_cache, chromosomal_holdouts
import pandas as pd

def test_chromosomal_holdouts_generator():
    x = pd.read_csv("test_dataset/x.csv", index_col=0)
    generator = holdouts_generator(
        x, x, x, 
        holdouts=chromosomal_holdouts([
            (
                [19],
                [([12],[([7], None)]), ([11],[([7], None)])]
            ),
            (
                [2],
                [([12],[([7], None)]), ([11],[([7], None)])]
            )
        ])
    )
    list(generator())
    list(generator())
    generator = holdouts_generator(
        x, x, x, 
        holdouts=chromosomal_holdouts([
            (
                [19],
                [([12],[([7], None)]), ([11],[([7], None)])]
            ),
            (
                [2],
                [([12],[([7], None)]), ([11],[([7], None)])]
            )
        ]),
        cache=True
    )
    list(generator())
    list(generator())
    clear_holdouts_cache()