import os
import pickle as pkl
from logging import DEBUG
from flwr.common.logger import log
import tensorflow as tf
from clients.base import Combination
from clients.tensorflow import TensorflowDatasetHandler, TensorflowModelHandler
from clients.utils.distributions import dataset_resizing

DATASET_DIR = "/data/tensorflow"




class TensorflowDatasetHandlerCifar(TensorflowDatasetHandler):
    dataset = ""
    _dataset = None

    @classmethod
    def load_data(cls, partition: int, nodes: int, training_set_size: int = 50_000, test_size: int = 10_000,
                  random: bool = False, distribution: str = 'flat', distribution_parameters: dict = {}):
        file = f"{DATASET_DIR}/{cls.dataset}.pkl"
        if cls._dataset is None:
            try:
                with open(file, "rb") as f:
                    cls._dataset = pkl.load(f)
            except FileNotFoundError:
                dataset_loader = getattr(tf.keras.datasets, cls.dataset, tf.keras.datasets.cifar10)
                cls._dataset = dataset_loader.load_data()
                if not os.path.exists(DATASET_DIR):
                    os.makedirs(DATASET_DIR)
                with open(file, "wb") as f:
                    pkl.dump(cls._dataset, f)
        ((x_train, y_train), (x_test, y_test)) = cls._dataset
        x_train = x_train[:training_set_size]
        y_train = y_train[:training_set_size]
        x_test = x_test[:test_size]
        y_test = y_test[:test_size]
        cls._dataset = ((x_train, y_train), (x_test, y_test))
        res = dataset_resizing(cls._dataset,
                     partition,
                     nodes,
                     distribution=distribution,
                     distribution_parameters=distribution_parameters)
        return res

class TensorflowCifar(Combination):
    model_handler = TensorflowModelHandler()
    dataset_handler = TensorflowDatasetHandlerCifar()

class TensorflowCifar100(TensorflowCifar):
    def __init__(self, **kwargs):
        Combination.__init__(self, **kwargs)
        cifar100 = tf.keras.applications.MobileNetV2((32, 32, 3), classes=100, weights=None)
        cifar100.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = cifar100
        self.dataset_handler.dataset = "cifar100"

class TensorflowCifar10(TensorflowCifar):
    def __init__(self, **kwargs):
        Combination.__init__(self, **kwargs)
        cifar10 = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        cifar10.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = cifar10
        self.dataset_handler.dataset = "cifar10"


