from holdouts_generator import cached_holdouts_generator, skip, store_keras_result, load_result, add_work_in_progress
from typing import Tuple, Dict
from keras.datasets import boston_housing
from .utils import example_random_holdouts, clear_all_cache
from keras import Sequential
from keras.layers import Dense
import time
import pytest

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
    start = time.time()
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
    return {
        "history": history,
        "x_train": training[0],
        "y_train_true": training[1],
        "x_test": testing[0],
        "y_test_true": testing[1],
        "model": model,
        "time": time.time() - start,
        "cache_dir":"holdouts",
        "results_directory":"results",
        "hyper_parameters": hyper_parameters,
        "parameters": parameters
    }


def test_keras_cache():
    clear_all_cache(results_directory="results", cache_dir="holdouts")

    (x_train, y_train), _ = boston_housing.load_data()
    generator = cached_holdouts_generator(
        x_train, y_train, holdouts=example_random_holdouts, cache_dir="holdouts", skip=skip)
    hyper_parameters = {
        "epochs": 1
    }

    for _, _, inner in generator(hyper_parameters):
        for _ in inner(hyper_parameters):
            pass

    for (training, testing), outer_key, inner in generator(results_directory="results", hyper_parameters=hyper_parameters):
        with pytest.raises(ValueError):
            load_result("results", outer_key, hyper_parameters)
        add_work_in_progress("results", outer_key, hyper_parameters)
        for (inner_training, inner_testing), inner_key, _ in inner(results_directory="results", hyper_parameters=hyper_parameters):
            store_keras_result(inner_key, **train(inner_training, inner_testing, hyper_parameters))
            with pytest.raises(ValueError):
                store_keras_result(inner_key, **train(inner_training, inner_testing, hyper_parameters))
        store_keras_result(outer_key, **train(training, testing, hyper_parameters))

    for (training, testing), outer_key, inner in generator(results_directory="results", hyper_parameters=hyper_parameters):
        assert training is None
        assert testing is None
        assert not inner()
        load_result("results", outer_key, hyper_parameters)

    clear_all_cache(results_directory="results", cache_dir="holdouts")
