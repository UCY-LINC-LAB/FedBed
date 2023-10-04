from clients.base import Combination
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from clients.pytorch import PyTorchModelHandler, PyTorchDatasetHandler
from clients.pytorch_mobilenetv2_cifar import MobileNetV2
from logging import DEBUG
from flwr.common.logger import log

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PyTorchCifarModelHandler(PyTorchModelHandler):

    @classmethod
    def train(cls, model, dataset, epochs, *args, **kwargs):
        """Train the model on the training set."""
        log(DEBUG, f"Training started for {epochs} epochs and {len(dataset.dataset)}")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in tqdm(dataset):
                optimizer.zero_grad()
                criterion(model(images), labels).backward()
                optimizer.step()
        return len(dataset.dataset)

    @classmethod
    def test(cls, model, dataset, *args, **kwargs):
        """Validate the model on the test set."""
        model.eval()
        test_loss: float = 0
        correct: int = 0
        num_test_samples: int = 0
        log(DEBUG, f"Test started for {len(dataset.dataset)}")

        with torch.no_grad():
            for data, target in dataset:
                num_test_samples += len(data)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= num_test_samples
        return test_loss, correct / num_test_samples, num_test_samples

class PyTorchCifarDatasetHandler(PyTorchDatasetHandler):
    dataset = None
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class PyTorchCifarDatasetHandlerMobileNetV2(PyTorchDatasetHandler):
    dataset = None
    trf = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


class PyTorchCifar(Combination):
    model = Model().to(DEVICE)
    model_handler = PyTorchCifarModelHandler()
    dataset_handler = PyTorchCifarDatasetHandler()

class PyTorchCifar10(PyTorchCifar):
    def __init__(self, **kwargs):
        Combination.__init__(self, **kwargs)
        self.dataset_handler.__class__.batch_size = self.batch_size
        self.dataset_handler.__class__.dataset = "cifar10"
        self.dataset = "cifar10"

class PyTorchCifar100(PyTorchCifar):
    def __init__(self, **kwargs):
        Combination.__init__(self, **kwargs)
        self.dataset_handler.__class__.batch_size = self.batch_size
        self.dataset_handler.__class__.dataset = "cifar100"
        self.dataset = "cifar100"

class PyTorchMobileNetV2Cifar10(PyTorchCifar10):
    def __init__(self, **kwargs):
        PyTorchCifar10.__init__(self, **kwargs)
        self.model = MobileNetV2()
        self.model.to(DEVICE)
        self.dataset_handler = PyTorchCifarDatasetHandlerMobileNetV2()
        self.dataset_handler.__class__.batch_size = self.batch_size
        self.dataset_handler.__class__.dataset = "cifar10"
        self.dataset = "cifar10"


class PyTorchMobileNetV2Cifar100(PyTorchCifar100):
    model = MobileNetV2().to(DEVICE)
    dataset_handler = PyTorchCifarDatasetHandlerMobileNetV2()


