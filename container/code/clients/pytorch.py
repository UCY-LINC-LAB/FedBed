import copy
from collections import OrderedDict
from logging import DEBUG
from flwr.common.logger import log

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from clients.base import ModelHandler, DatasetHandler
from clients.utils.distributions import dataset_resizing
DATASET_DIR = "/data/pytorch"


class PyTorchModelHandler(ModelHandler):

    @classmethod
    def set_model_parameters(cls, model, parameters, *args, **kwargs):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    @classmethod
    def get_model_parameters(cls, model, *args, **kwargs):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

class PyTorchDatasetHandler(DatasetHandler):

    dataset = None
    trainset = None
    testset = None
    trf = None

    @classmethod
    def load_data(cls,
                  partition: int,
                  nodes: int,
                  training_set_size: int = 50_000,
                  test_size: int = 10_000,
                  random: bool = False,
                  distribution: str = 'flat',
                  distribution_parameters: dict = {}):
        
        DatasetLoaderClass = getattr(datasets, cls.dataset.upper(), datasets.CIFAR10)
        if cls.trainset is None or cls.testset is None:
            cls.trainset = DatasetLoaderClass(f"{DATASET_DIR}/{cls.dataset}", train=True, download=True, transform=cls.trf)
            cls.testset = DatasetLoaderClass(f"{DATASET_DIR}/{cls.dataset}", train=False, download=True, transform=cls.trf)
            (x_train, y_train), (x_test, y_test) = (cls.trainset.data, cls.trainset.targets), (cls.testset.data, cls.testset.targets)
            x_train = x_train[:training_set_size]
            y_train = y_train[:training_set_size]
            x_test = x_test[:test_size]
            y_test = y_test[:test_size]
            (cls.trainset.data, cls.trainset.targets), (cls.testset.data, cls.testset.targets) = (x_train, y_train), (x_test, y_test)

        trainset, testset = dataset_resizing((copy.deepcopy(cls.trainset), copy.deepcopy(cls.testset)), partition,
            nodes, distribution=distribution, distribution_parameters=distribution_parameters)
        
        log(DEBUG, trainset)      
        log(DEBUG, testset) 
        log(DEBUG, cls.batch_size)       
  
        return DataLoader(trainset, batch_size=cls.batch_size, shuffle=True), DataLoader(testset)