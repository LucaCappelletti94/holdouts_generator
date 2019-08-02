from holdouts_generator import clear_cache, cached_holdouts_generator, skip, store_keras_result, load_result, delete_results
from typing import Tuple, Dict
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


def train(training: Tuple, testing: Tuple, hyper_parameters: Dict):
    model = mlp()
    parameters = {
        "shuffle": True
    }
    history = model.fit(
        *training,
        **hyper_parameters,
        **parameters,
        validation_data=testing,
        verbose=0
    ).history
    return history, testing[0], testing[1], model, hyper_parameters, parameters


def test_keras_cache():
    (x_train, y_train), _ = boston_housing.load_data()
    generator = cached_holdouts_generator(
        x_train, y_train, holdouts=example_random_holdouts, skip=skip)
    hyper_parameters = {
        "epochs": 1
    }

    for _, _, inner in generator(hyper_parameters):
        for _ in inner(hyper_parameters):
            pass

    for (training, testing), outer_key, inner in generator(hyper_parameters):
        for (inner_training, inner_testing), inner_key, _ in inner(hyper_parameters):
            if inner_training is not None:
                store_keras_result(
                    inner_key, *train(inner_training, inner_testing, hyper_parameters))
        if training is not None:
            store_keras_result(
                outer_key, *train(training, testing, hyper_parameters))

    for _, outer_key, inner in generator(hyper_parameters):
        for _, inner_key, _ in inner(hyper_parameters):
            load_result(inner_key, hyper_parameters)
        load_result(outer_key, hyper_parameters)

    clear_cache()
    delete_results()
