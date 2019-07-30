from holdouts_generator import holdouts_generator, clear_cache, random_holdouts, cached_holdouts_generator, skip, store_keras_result, load_result, delete_results
import numpy as np
from typing import Tuple
from keras.datasets import boston_housing
from .utils import example_random_holdouts
from keras import Sequential
from keras.layers import Dense

def mlp()->Sequential:
    model = Sequential([
        *[Dense(units=10) for kwargs in range(2)],
        Dense(1, activation="relu"),
    ])
    model.compile(
        optimizer="nadam",
        loss="mse"
    )
    return model

def train(training:Tuple, testing:Tuple):
    model = mlp()
    history = model.fit(
        *training,
        epochs=1,
        validation_data=testing,
        verbose=0
    ).history
    return history, testing[0], testing[1], model

def test_keras_cache():
    (x_train, y_train), _ = boston_housing.load_data()
    generator = cached_holdouts_generator(x_train, y_train, holdouts=example_random_holdouts, skip=skip)

    for _, _, inner in generator():
        for _ in inner():
            pass

    for (training, testing), outer_key, inner in generator():            
        for (inner_training, inner_testing), inner_key, _ in inner():
            if inner_training is not None:
                store_keras_result(inner_key, *train(inner_training, inner_testing))
        if training is not None:
            store_keras_result(outer_key, *train(training, testing))
    
    for data, outer_key, inner in generator():
        for _, inner_key, _ in inner():
            load_result(inner_key)
        load_result(outer_key)

    clear_cache()
    delete_results()