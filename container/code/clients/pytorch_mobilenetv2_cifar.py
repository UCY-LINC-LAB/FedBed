from clients.base import Combination
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from logging import DEBUG
from flwr.common.logger import log

from clients.pytorch import PyTorchModelHandler, PyTorchDatasetHandler

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
class MobileNetV2(nn.Module):
    def __init__(self, class_num=10, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained

        # load model #####################################
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'mobilenet_v2',
                                    pretrained=self.pretrained)

        # set last layer ################################
        if self.model.classifier[1].out_features != self.class_num:
            self.model.classifier[1] = nn.Linear(1280, self.class_num)
        # self.model = self.model.to(device)
        log(DEBUG, f"MobileNetV2 is initialized!!!")

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out


