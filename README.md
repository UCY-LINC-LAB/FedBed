# Container

Reference container used in the experiments. 

## Env Variables

#### FL Server Parameters

| Variable                 |                         Description                          | Default Value       |                            Values                            |
| ------------------------ | :----------------------------------------------------------: | ------------------- | :----------------------------------------------------------: |
| FL_STRATEGY              |            The aggregation strategy of the Server            | FedAvg              | “FedAvg”,  "FastAndSlow",  "FaultTolerantFedAvg",  "FedAdagrad",  "FedAdam",  "FedAvg", "FedAvgAndroid", "FedAvgM",  "FedFSv0",  "FedFSv1",  "FedYogi", "QFedAvg" |
| FL_NUM_OF_ROUNDS         |                       Number of rounds                       | 3                   |                                                              |
| FL_FRACTION_FIT          |             Fraction of clients used during training         | 0.1, aka sample 10% |                                                              |
| FL_FRACTION_EVAL         |           Fraction of clients used during validation         | 0.1                 |                                                              |
| FL_MIN_EVAL_CLIENTS      |       Minimum number of clients used during validation       | 2                   |                                                              |
| FL_MIN_FIT_CLIENTS       |       Minimum number of clients used during training         | 2                   |                                                              |
| FL_MIN_AVAILABLE_CLIENTS | Minimum number of clients that need to be connected to the server before a training round can start | 2                   |                                                              |
| FL_EVAL_DATASET          | If the server will evaluate the results locally after every round | "false"             | "true" or "false" |
| FL_DATASET               | The selected dataset (It will be taken into account only if previous parameter is false) | CIFAR10 | CIFAR10 (tensorflow/pytorch), CIFAR100(tensorflow), MNIST (pytorch/tensorflow/pytorch_light/sklearn/mxnet) |
| FL_BACKEND               | This should be set only if `FL_EVAL_DATASET` is true | - | pytorch/tensorflow/pytorch_light/sklearn/mxnet (depending on the dataset) |
| FL_TRAINING_SET_SIZE | the size of the training set (This could be set only if `FL_EVAL_DATASET` is true) | -1            |  max values: MNIST = 60000, CIFAR10 = 50000                      |
| FL_TEST_SET_SIZE | the size of the testing set (This could be set only if `FL_EVAL_DATASET` is true) | -1            | max values: MNIST = 10000, CIFAR10 = 10000                         |
| FL_PROFILE_PREFIX | prefix for the profiler in order to separate different runs/logs | "" |  |

*Since different strategies may have different parameters, we can also introduce (in next version) specific parameters for specific strategies (e.g. coefficients, parameters, etc.) if it is needed*



#### FL Client Parameters

| Variable          | Description                                                  | Default Value | Values                  |
| ----------------- | ------------------------------------------------------------ | ------------- | ----------------------- |
| FL_BACKEND        | the backend ML implementation                                | pytorch       | pytorch/tensorflow/pytorch_light/sklearn/mxnet (depending on the dataset)|
| FL_NUM_OF_THREADS | the threads that will be occupied by the client for training | 0             |  (not tested)                       |
| FL_SERVER         | the hostname of the server to connect to                     | "[::]"        |                         |
| FL_DATASET        | the dataset to be used for the training                      | CIFAR10       | CIFAR10 (tensorflow/pytorch),CIFAR100(tensorflow), MNIST (pytorch/tensorflow/pytorch_light/sklearn/mxnet) |
| FL_TRAINING_SET_SIZE | the size of the training set                              | -1            |  max values: MNIST = 60000, CIFAR10 = 50000                      |
| FL_TEST_SET_SIZE | the size of the testing set                                   | -1            | max values: MNIST = 10000, CIFAR10 = 10000                         |
| FL_PROFILE_PREFIX | prefix for the profiler in order to separate different runs/logs | "" |  |
| FL_EPOCHS | how many epochs a model will be trained (for deep learning models)   | 1             | positive integers |
| FL_HOST | The hostname of the node (optional) | "" |  |
| FL_NODES | The number of the clients | - | positive integer |
| FL_NODE_ID | A node's identifier which is an integer from 0 to `FL_NODES`-1 | - | positive integer |
| FL_DATASET_DISTRIBUTION | The data distribution on nodes | "flat" | "gaussian", "pareto", "flat", "dirichlet" |
| FL_DATASET_DISTRIBUTION_PARAMETERS | A JSON object with the parameters of distributions | "{}" | A JSON object | 
| FL_DATASET_RANDOM | Specify if the dataset is IID or not | False | True/False |


*Internal timings from services will be found at `/profile/` and the datasets should be at `/data/`. If the datasets are not at `/data/`, the framework will download them. Furthermore, the timings will be stored as json-lines in the `/profile/profile.jl` *
