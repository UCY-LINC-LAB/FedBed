from collections import OrderedDict

import clients
from typing import Callable, Optional, Tuple
import torchvision
import flwr as fl
import os
import torch
from torchvision.datasets import cifar

from clients import mxnet_mnist

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_eval_fn():
    """Return an evaluation function for centralized evaluation."""

    def evaluate(parameters):
        backend = str(os.getenv("FL_BACKEND")).lower()
        dataset = os.getenv("FL_DATASET", 'CIFAR10')
        training_set_size = int(os.getenv("FL_TRAINING_SET_SIZE", -1))
        test_set_size = int(os.getenv("FL_TEST_SET_SIZE", -1))
        dataset = dataset.lower()
        backend = backend if backend in ["pytorch", "tensorflow", "mxnet", "pytorch_light", "sklearn"] else "pytorch"
        backend_class = f"{backend}_{dataset}"
        Module = getattr(clients, backend_class)
        print("Evaluation started")
        batch_size=1
        epochs=1
        module = Module(dataset=dataset, batch_size=batch_size, epochs=epochs)
        print(f"Module {backend_class} loaded")

        model = module.model
        print(f"Model {model} loaded")
        _, testloader = module.dataset_handler.load_data(0, 1, test_size=test_set_size, training_set_size=training_set_size)
        print(f"Dataset is loaded")

        module.model_handler.set_model_parameters(model, parameters)
        print(f"Evaluation model updated")

        loss, accuracy, num_of_res = module.model_handler.test(model, testloader)
        print(f"Test is performed {loss}, {accuracy}, {num_of_res}")

        return loss, {"accuracy": accuracy}


    return evaluate