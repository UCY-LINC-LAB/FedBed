from random import randint

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np
from sklearn.model_selection import train_test_split
from logging import DEBUG
from flwr.common.logger import log
from clients.base import ModelHandler, DatasetHandler, Combination
from clients.utils.distributions import dataset_resizing

DATASET_DIR = "/data/sklearn"

class SklearnModelHandler(ModelHandler):

    @classmethod
    def set_model_parameters(cls, model, parameters, *args, **kwargs):
        model.coef_ = parameters[0]
        if model.fit_intercept:
            model.intercept_ = parameters[1]
        return model


    @classmethod
    def get_model_parameters(cls, model, *args, **kwargs):
        if model.fit_intercept:
            params = (model.coef_, model.intercept_)
        else:
            params = (model.coef_,)
        return params

    @classmethod
    def train(cls, model, dataset, *args, **kwargs):
        x_train, y_train = dataset
        model.fit(x_train, y_train)
        return len(y_train)

    @classmethod
    def test(cls, model, dataset, *args, **kwargs):
        x_test, y_test = dataset
        loss = log_loss(y_test, model.predict_proba(x_test))
        accuracy = model.score(x_test, y_test)
        return loss, accuracy, len(x_test)

class SklearnDatasetHandler(DatasetHandler):

    _dataset = None

    @classmethod
    def load_data(cls,
                  partition: int,
                  nodes: int,
                  training_set_size: int = 50_000,
                  test_size: int = 10_000,
                  random: bool = False,
                  distribution: str = 'flat',
                  distribution_parameters: dict = {}):

        if cls._dataset is None:
            cls._dataset = fetch_openml('mnist_784', data_home=DATASET_DIR, version=1)
        x, y = cls._dataset['data'], cls._dataset['target']
        y = y.astype(np.uint8)
        seed = randint(0, 10_000) if not random else 0
        X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                            train_size=training_set_size,
                                                            test_size=test_size,
                                                            random_state=seed)

        (_X_train, _y_train), (_X_test, _y_test) = dataset_resizing(
            ((X_train, y_train),(X_test, y_test)),
            partition, nodes,
            distribution=distribution,
            distribution_parameters=distribution_parameters)

        return (_X_train, _y_train), (_X_test, _y_test)


def _set_initial_params(model):
    n_classes = 10  # MNIST has 10 classes
    n_features = 784  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
    return model

model = LogisticRegression(
            penalty="l2",
            warm_start=True,  # prevent refreshing weights when fitting
            max_iter=1
            )

model = _set_initial_params(model)


class SklearnMnist(Combination):
    model_handler = SklearnModelHandler()
    dataset_handler = SklearnDatasetHandler()
    model = model


