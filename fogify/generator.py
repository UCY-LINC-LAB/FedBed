from dataclasses import dataclass
from string import Template

DATASETS_AND_MODELS = dict(MNIST=["pytorch", "tensorflow", "pytorch_light", "sklearn", "mxnet"],
    CIFAR10=["pytorch", "tensorflow"], CIFAR100=["pytorch", "tensorflow"])


@dataclass
class DockerComposeGenerator:
    docker_compose_filename: str
    num_of_clients: int
    strategy: str = "FedAvg"
    num_of_rounds: int = 100
    central_eval: str = "true"
    eval_clients: int = None
    fit_clients: int = None
    backend: str = "pytorch"
    dataset: str = "CIFAR100"
    code_folder: str = "/home/ubuntu/research/FDLC/container/code/"
    data_folder: str = "/home/ubuntu/research/FDLC/fogify"
    proxy: str = "http://proxy.cs.ucy.ac.cy:8008/"
    server_test_size: int = 10_000
    server_training_size: int = 50_000
    test_size: int = 10_000
    train_size: int = 50_000
    epochs: int = 1
    dataset_distribution: str = 'flat'
    dataset_distribution_parameters: str = '{}'
    

    def __post_init__(self):
        if self.eval_clients is None:
            self.eval_clients = self.num_of_clients
        if self.fit_clients is None:
            self.fit_clients = self.num_of_clients

    docker_compose_file = """
    version: '3.7'
    services:
    """

    server_template = """
      server:
        image: fl_chaos
        command: [ "python", "/code/server.py"]
        environment:
          FL_STRATEGY: $strategy
          # "FastAndSlow", "FaultTolerantFedAvg", "FedAdagrad", "FedAdam", "FedAvg",
          # "FedAvgAndroid", "FedAvgM", "FedFSv0", "FedFSv1", "FedYogi", "QFedAvg"
          FL_NUM_OF_ROUNDS: $num_of_rounds
          FL_FRACTION_FIT: 0.1
          FL_FRACTION_EVAL: 0.1
          FL_MIN_EVAL_CLIENTS: $eval_clients
          FL_MIN_FIT_CLIENTS: $fit_clients
          FL_MIN_AVAILABLE_CLIENTS: $eval_clients
          FL_PROFILE_PREFIX: $backend-$dataset-$strategy-$central_eval-$eval_clients-2023-small
          FL_BACKEND: $backend
          FL_DATASET: $dataset
          FL_EVAL_DATASET: "$central_eval"
          HTTP_PROXY: $proxy
          HTTPS_PROXY: $proxy
          https_proxy: $proxy
          http_proxy: $proxy
          FL_TEST_SET_SIZE: $test_size
          FL_TRAINING_SET_SIZE: $train_size
          FL_EPOCHS: $epochs
        volumes:
          - $code_folder:/code/
          - $data_folder/profile/server/:/profile/
          - $data_folder/data/:/data/
        expose:
        - 8080
    """
#          FL_TEST_SET_SIZE: $test_size

    client_template = """
      client_$client_id:
        image: fl_chaos
        depends_on:
          - server
        command: 'sh -c "sleep 20 && python /code/client.py"'
        environment:
          FL_SERVER: server
          FL_BACKEND: $backend
          FL_DATASET: $dataset
          FL_NODES: $nodes
          FL_NODE_ID: $client_id
          FL_DATASET_DISTRIBUTION: $dataset_distribution
          FL_DATASET_RANDOM: 'False'
          FL_DATASET_DISTRIBUTION_PARAMETERS: '$dataset_distribution_parameters'
          FL_PROFILE_PREFIX: $client_id-$backend-$dataset-$eval_clients-2023-small
          HTTP_PROXY: $proxy
          HTTPS_PROXY: $proxy
          https_proxy: $proxy
          http_proxy: $proxy
          FL_TRAINING_SET_SIZE: $train_size
          FL_TEST_SET_SIZE: $test_size
          FL_EPOCHS: 1

        volumes:
          - $code_folder:/code/
          - $data_folder/profile/clients/:/profile/
          - $data_folder/data/:/data/

    """

    def generate_docker_compose(self):
        res = self.docker_compose_file
        res += Template(self.server_template).substitute(strategy=self.strategy, num_of_rounds=self.num_of_rounds,
                                                         central_eval=self.central_eval, eval_clients=self.eval_clients,
                                                         fit_clients=self.fit_clients, backend=self.backend,
                                                         dataset=self.dataset, code_folder=self.code_folder,
                                                         data_folder=self.data_folder, proxy=self.proxy,
                                                         train_size=self.server_training_size, test_size=self.server_test_size, epochs=self.epochs)

        for i in range(self.num_of_clients):
            res += Template(self.client_template).substitute(
                client_id=i, dataset=self.dataset, backend=self.backend,
                code_folder=self.code_folder, data_folder=self.data_folder, proxy=self.proxy,
                nodes=self.num_of_clients, dataset_distribution=self.dataset_distribution,
                train_size=self.train_size, test_size=self.test_size, epochs=self.epochs, dataset_distribution_parameters=self.dataset_distribution_parameters,
                eval_clients=self.eval_clients)
        with open(f"{self.docker_compose_filename}", "w") as f:
            f.write(res)


