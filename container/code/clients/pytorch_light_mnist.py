from collections import OrderedDict
from torch import nn
from torch.nn import functional as F

import torch
import pytorch_lightning as pl
from torchvision.transforms import ToTensor

from clients.base import ModelHandler, Combination
from clients.pytorch import PyTorchDatasetHandler


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage=None):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)

class PytorchLightModelHandler(ModelHandler):

    @staticmethod
    def _get_parameters(model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    @staticmethod
    def _set_parameters(model, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    @classmethod
    def set_model_parameters(cls, model, parameters, *args, **kwargs):
        cls._set_parameters(model.encoder, parameters[:4])
        cls._set_parameters(model.decoder, parameters[4:])


    @classmethod
    def get_model_parameters(cls, model, *args, **kwargs):
        encoder_params = cls._get_parameters(model.encoder)
        decoder_params = cls._get_parameters(model.decoder)
        return encoder_params + decoder_params

    @classmethod
    def train(cls, model, dataset, epochs, testloader, *args, **kwargs):
        train_loader, test_loader = dataset, testloader
        trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=0)
        trainer.fit(model, train_loader, test_loader)
        return len(train_loader.dataset)


    @classmethod
    def test(cls, model, dataset, *args, **kwargs):
        trainer = pl.Trainer(progress_bar_refresh_rate=0)
        results = trainer.test(model, dataset)
        loss = results[0]["test_loss"]
        accuracy = 0.0  # TODO find a way to check accuracy
        return loss, accuracy, len(dataset.dataset)

class PytorchLightDatasetHandler(PyTorchDatasetHandler):
    dataset = "mnist"
    trf = ToTensor()

class PytorchLightMnist(Combination):
    model_handler = PytorchLightModelHandler()
    dataset_handler = PytorchLightDatasetHandler()
    model = LitAutoEncoder()

    def __init__(self, **kwargs):
        Combination.__init__(self, **kwargs)
        self.dataset_handler.__class__.batch_size = self.batch_size
        self.dataset_handler.__class__.dataset = "mnist"
        self.dataset = "mnist"

