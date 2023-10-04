import json
import os
import numpy as np
os.unsetenv("http_proxy")
os.unsetenv("https_proxy")

import flwr as fl
# from flwr.client import start_client
# from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from logging import DEBUG
from flwr.common.logger import log

import clients
from clients.base import FLClient

backend = str(os.getenv("FL_BACKEND")).lower()
num_of_threads = int(os.getenv("FL_NUM_OF_THREADS", 0))
dataset = os.getenv("FL_DATASET", 'CIFAR10')
server = os.getenv("FL_SERVER", "[::]")
training_set_size = int(os.getenv("FL_TRAINING_SET_SIZE", -1))
test_set_size = int(os.getenv("FL_TEST_SET_SIZE", -1))
epochs = int(os.getenv("FL_EPOCHS", 1))
host = os.getenv("FL_HOST", "")

# new parameters for distribution
num_of_nodes = int(os.getenv("FL_NODES"))
node_id = int(os.getenv("FL_NODE_ID"))
distribution = str(os.getenv("FL_DATASET_DISTRIBUTION", 'flat'))
distribution_randomness = bool(os.getenv("FL_DATASET_RANDOM", False))  # TODO Add random as functionality
distribution_params_str = str(os.getenv("FL_DATASET_DISTRIBUTION_PARAMETERS", ''))

distribution_params = {}
try:
    distribution_params = json.loads(distribution_params_str)
except Exception:
    pass
dataset_params = dict(
    partition=node_id,
    nodes=num_of_nodes,
    test_size=test_set_size,
    training_set_size=training_set_size,
    random=distribution_randomness,
    distribution=distribution,
    distribution_parameters=distribution_params
    )

if __name__ == "__main__":
    dataset = dataset.lower()
    backend = backend if backend in ["pytorch", "pytorch_mobilenetv2", "tensorflow", "mxnet", "pytorch_light", "sklearn"] else "pytorch"
    backend_class = f"{backend}_{dataset}"
    handler = getattr(clients, backend_class)
    log(DEBUG, f"{dataset} {backend_class} {handler}")
    if handler:
        fl.client.start_numpy_client(server_address=server + ":8080", client=FLClient(handler=handler,
                                                                       dataset=dataset,
                                                                       dataset_params=dataset_params,
                                                                       epochs=epochs,
                                                                       host=host
                                                                       ))
    else:
        raise ValueError(f"The combination of {backend} and {dataset} does not exist")