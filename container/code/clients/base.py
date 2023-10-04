import os
from abc import ABC
from dataclasses import dataclass
from typing import Any
from flwr.common import EvaluateRes, EvaluateIns, parameters_to_ndarrays
from logging import DEBUG
from flwr.common.logger import log

from clients.utils.profiler import Profiler

class DatasetHandler(ABC):

    @classmethod
    def load_data(cls,
                  partition: int,
                  nodes: int,
                  training_set_size: int = 50_000,
                  test_size: int = 10_000,
                  random: bool = False,
                  distribution: str = 'flat',
                  distribution_parameters: dict = {}):
        pass

class ModelHandler(ABC):

    set_parameters_on_fit: bool = True

    @classmethod
    def set_model_parameters(cls, model, parameters, *args, **kwargs):
        pass

    @classmethod
    def get_model_parameters(cls, model, *args, **kwargs):
        pass

    @classmethod
    def train(cls, model, dataset, *args, **kwargs):
        pass

    @classmethod
    def test(cls, model, dataset, *args, **kwargs):
        pass

@dataclass
class Combination(object):
    model: Any
    dataset_handler: DatasetHandler
    model_handler: ModelHandler
    
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])



import flwr as fl

class FLClient(fl.client.NumPyClient):

    handler: Combination = None

    def __init__(self,
                 handler: Combination,
                 dataset: str,
                 dataset_params: dict,
                 batch_size: int = 32,
                 epochs: int = 1,
                 host: str = "",
                 *args, **kwargs):
        self.handler = handler(dataset=dataset, batch_size=batch_size, epochs=epochs, **kwargs)
        self.dataset_params = dataset_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.handler.model
        self.host = host

    def get_properties(self, config):
        log(DEBUG, f"get_properties")
        return {"hostname": self.host}

    @Profiler("load_data")
    def load_data(self):
        log(DEBUG, f"load_data")

        return self.handler.dataset_handler.load_data(**self.dataset_params)

    @Profiler("get_parameters", log_size=True)
    def get_parameters(self, config=None):
        log(DEBUG, f"get_parameters")

        return self.handler.model_handler.get_model_parameters(self.model)

    @Profiler("set_parameters")
    def set_parameters(self, parameters):
        log(DEBUG, f"set_parameters")

        self.handler.model_handler.set_model_parameters(self.model, parameters)

    @Profiler("train")
    def train(self):
        log(DEBUG, f"train")

        trainloader, testloader = self.load_data()
        # log(DEBUG, f"train SIZE:  {len(testloader[0])} {len(testloader[1])} {len(trainloader[0])} {len(trainloader[1])}")
        return self.handler.model_handler.train(self.model, trainloader, testloader=testloader, epochs=self.epochs, batch_size=self.batch_size)

    @Profiler("test")
    def test(self):
        log(DEBUG, f"test")

        _, testloader = self.load_data()
        # log(DEBUG, f"test SIZE:  {len(testloader[0])} {len(testloader[1])} {len(_[0])} {len(_[1])}")
        return self.handler.model_handler.test(self.model, testloader)

    def fit(self, parameters, config):
        log(DEBUG, f"fit")

        if self.handler.model_handler.set_parameters_on_fit:
            self.set_parameters(parameters)
        num_of_examples = self.train()
        res = list(self.get_parameters()), num_of_examples, {}
        Profiler.log_metrics()
        return res

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, num_examples = self.test()
        Profiler.log_metric("prev_accuracy", accuracy)
        return loss, num_examples, {"accuracy": accuracy}#, Profiler.profiles

Profiler.profiles
def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        loss, num_examples, accuracy, metrics = self.numpy_client.evaluate(parameters, ins.config)
        if type(accuracy) == dict:
            accuracy = accuracy.get("accuracy", 0.0)
        metrics["accuracy"] = accuracy
        return EvaluateRes(loss=loss, num_examples=num_examples, metrics=metrics)