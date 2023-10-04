import copy
import gzip
import struct

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from clients.base import ModelHandler, DatasetHandler, Combination
from logging import DEBUG
from flwr.common.logger import log
# Fixing the random seed
from clients.utils.distributions import dataset_resizing

mx.random.seed(42)

DATASET_DIR = "/data/mxnet"

DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]

class MxnetModelHandler(ModelHandler):

    @classmethod
    def set_model_parameters(cls, model, parameters, *args, **kwargs):
        params = zip(model.collect_params(".*weight").keys(), parameters)
        for key, value in params:
            model.collect_params().setattr(key, value)


    @classmethod
    def get_model_parameters(cls, model, *args, **kwargs):
        param = []
        for val in model.collect_params(".*weight").values():
            p = val.data()
            param.append(p.asnumpy())
        return param

    @classmethod
    def train(cls, model, dataset, epochs, *args, **kwargs):
        trainer = gluon.Trainer(model.collect_params(), "sgd", {"learning_rate": 0.01})
        accuracy_metric = mx.metric.Accuracy()
        loss_metric = mx.metric.CrossEntropy()
        metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        for i in range(epochs):
            dataset.reset()
            num_examples = 0
            for batch in dataset:
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=DEVICE, batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=DEVICE, batch_axis=0)
                outputs = []
                with ag.record():
                    for x, y in zip(data, label):
                        z = model(x)
                        loss = softmax_cross_entropy_loss(z, y)
                        loss.backward()
                        outputs.append(z.softmax())
                        num_examples += len(x)
                metrics.update(label, outputs)
                trainer.step(batch.data[0].shape[0])
            trainings_metric = metrics.get_name_value()
            print("Accuracy & loss at epoch %d: %s" % (i, trainings_metric))
        return num_examples

    @classmethod
    def test(cls, model, dataset, *args, **kwargs):
        accuracy_metric = mx.metric.Accuracy()
        loss_metric = mx.metric.CrossEntropy()
        metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        dataset.reset()
        num_examples = 0
        for batch in dataset:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=DEVICE, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=DEVICE, batch_axis=0)
            outputs = []
            for x in data:
                outputs.append(model(x).softmax())
                num_examples += len(x)
            metrics.update(label, outputs)
        metrics.update(label, outputs)
        [accuracy, loss] = metrics.get_name_value()
        return loss[1], accuracy[1], num_examples



class MxnetDatasetHandler(DatasetHandler):

    dataset = None
    batch_size = 1
    train_dataset = None
    test_dataset = None

    @classmethod
    def load_data(cls,
                  partition: int,
                  nodes: int,
                  training_set_size: int = 50_000,
                  test_size: int = 10_000,
                  random: bool = False,
                  distribution: str = 'flat',
                  distribution_parameters: dict = {}):
        def read_data(label_url, image_url):
            with gzip.open(mx.test_utils.download(label_url, dirname=DATASET_DIR)) as flbl:
                struct.unpack(">II", flbl.read(8))
                label = np.frombuffer(flbl.read(), dtype=np.int8)
            with gzip.open(mx.test_utils.download(image_url, dirname=DATASET_DIR), 'rb') as fimg:
                _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
                image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
                image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32) / 255
            return (label, image)

        print("Read Dataset")
        path = 'http://data.mxnet.io/data/mnist/'
        if cls.train_dataset is None or cls.test_dataset is None:
            (y_train, x_train) = read_data(path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
            (y_test, x_test) = read_data(path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')
            x_train = x_train[:training_set_size]
            y_train = y_train[:training_set_size]
            x_test = x_test[:test_size]
            y_test = y_test[:test_size]
            cls.train_dataset = (y_train, x_train)
            cls.test_dataset = (y_test, x_test)

        (y_train, x_train), (y_test, x_test) = dataset_resizing(
                                                        (copy.deepcopy(cls.train_dataset), copy.deepcopy(cls.test_dataset)), partition,
            nodes, distribution=distribution, distribution_parameters=distribution_parameters)

        try:
            train_data = mx.io.NDArrayIter(x_train, y_train, cls.batch_size, shuffle=True)
            val_data = mx.io.NDArrayIter(x_test, y_test, cls.batch_size)
        except Exception as ex:
            print(f"{ex}")

        return train_data, val_data

def MXNet():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(64, activation="relu"))
    net.add(nn.Dense(10))
    net.collect_params().initialize()
    return net


model = MXNet()
init = nd.random.uniform(shape=(2, 784))
model(init)

class MxnetMnist(Combination):
    model_handler = MxnetModelHandler()
    dataset_handler = MxnetDatasetHandler()
    model = model


