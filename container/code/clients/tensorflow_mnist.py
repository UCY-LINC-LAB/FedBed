import tensorflow as tf
from clients.base import Combination
from clients.tensorflow import TensorflowDatasetHandler, TensorflowModelHandler
from clients.utils.distributions import dataset_resizing

from logging import DEBUG
from flwr.common.logger import log
DATASET_DIR = "/data/tensorflow"

class TensorflowDatasetHandlerMnist(TensorflowDatasetHandler):
    dataset = "mnist"
    _dataset = None

    @classmethod
    def load_data(cls, partition: int, nodes: int, training_set_size: int = 50_000, test_size: int = 10_000,
                  random: bool = False, distribution: str = 'flat', distribution_parameters: dict = {}):
        file = f"{DATASET_DIR}/{cls.dataset}.npz"
        if cls._dataset is None:
            cls._dataset = tf.keras.datasets.mnist.load_data(file)

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

class TensorflowMnist(Combination):
    model_handler = TensorflowModelHandler()
    dataset_handler = TensorflowDatasetHandlerMnist()

    def __init__(self, **kwargs):
        Combination.__init__(self, **kwargs)
        # Build and compile Keras model
        # model = tf.keras.models.Sequential(
        #     [tf.keras.layers.Flatten(input_shape=(28, 28)),
        #      tf.keras.layers.Dense(128, activation="relu"),
        #      tf.keras.layers.Dropout(0.2),
        #      tf.keras.layers.Dense(10, activation="softmax"), ])

        model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ])

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = model