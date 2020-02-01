from holdouts_generator import holdouts_generator, balanced_random_holdouts
import pandas as pd
import numpy as np


def test_bed_files():
    bed = pd.DataFrame({
        "chrom": ["chr1"]*100,
        "chromStart": range(100),
        "chromEnd": range(100, 200)
    })

    classes = np.random.randint(2, size=(len(bed), 1))

    generator = holdouts_generator(bed, classes, holdouts=balanced_random_holdouts(
        test_sizes=[0.3],
        quantities=[1],
        random_state=42
    ))
    for (training, testing), _ in generator():
        assert isinstance(training[0], pd.DataFrame)
        assert isinstance(testing[0], pd.DataFrame)
