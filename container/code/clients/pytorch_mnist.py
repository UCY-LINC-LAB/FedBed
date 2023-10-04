from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from logging import DEBUG
from flwr.common.logger import log
from clients.base import Combination
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from clients.pytorch import PyTorchModelHandler, PyTorchDatasetHandler

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MNISTNet(nn.Module):
    """Simple CNN adapted from Pytorch's 'Basic MNIST Example'."""

    def __init__(self) -> None:
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass.
        Parameters
        ----------
        x: Tensor
            Mini-batch of shape (N,28,28) containing images from MNIST dataset.
        Returns
        -------
        output: Tensor
            The probability density of the output being from a specific class given the input.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class PyTorchMnistModelHandler(PyTorchModelHandler):

    @classmethod
    def train(cls, model, dataset, epochs, *args, **kwargs):
        """Train the model on the training set."""
        log(DEBUG, "Training Started!!!")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in tqdm(dataset):
                optimizer.zero_grad()
                criterion(model(images), labels).backward()
                optimizer.step()
                log(DEBUG, "Training Continue!!!")

        log(DEBUG, "Training Finished!!!")
        return len(dataset.dataset)

    # @classmethod
    # def train(cls, model, dataset, epochs, *args, **kwargs):
    #     """Train the model on the training set."""
    #     model.train()
    #     optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    #     scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    #     print(f"Training {epochs} epoch(s) w/ {len(dataset)} mini-batches each")
    #     for epoch in range(epochs):  # loop over the dataset multiple times
    #         loss_epoch: float = 0.0
    #         num_examples_train: int = 0
    #         for batch_idx, (data, target) in enumerate(dataset):
    #             # Grab mini-batch and transfer to device
    #             # data, target = data.to(device), target.to(device)
    #             num_examples_train += len(data)

    #             # Zero gradients
    #             optimizer.zero_grad()

    #             output = model(data)
    #             loss = F.nll_loss(output, target)
    #             loss.backward()
    #             optimizer.step()

    #             loss_epoch += loss.item()
    #             if batch_idx % 10 == 8:
    #                 print("Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}\t\t\t\t".format(epoch, num_examples_train,
    #                     len(dataset) * dataset.batch_size,
    #                     100.0 * num_examples_train / len(dataset) / dataset.batch_size, loss.item(), ),
    #                     end="\r", flush=True, )
    #         scheduler.step()
    #     return num_examples_train

    @classmethod
    def test(cls, model, dataset, *args, **kwargs):
        """Validate the model on the test set."""
        model.eval()
        test_loss: float = 0
        correct: int = 0
        num_test_samples: int = 0
        log(DEBUG, f"Test Started!!! {dataset}")

        with torch.no_grad():
            for data, target in dataset:
                num_test_samples += len(data)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= num_test_samples
        return test_loss, correct / num_test_samples, num_test_samples

class PyTorchMnistDatasetHandler(PyTorchDatasetHandler):
    dataset = "mnist"
    trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

class PyTorchMnist(Combination):
    

    def __init__(self, **kwargs):
        Combination.__init__(self, **kwargs)
        self.model = MNISTNet().to(DEVICE)
        self.model_handler = PyTorchMnistModelHandler()
        self.dataset_handler = PyTorchMnistDatasetHandler()
        self.dataset_handler.__class__.batch_size = self.batch_size
        self.dataset_handler.__class__.dataset = "mnist"
        self.dataset = "mnist"




