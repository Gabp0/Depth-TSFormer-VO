import os
import glob
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
from json import dump
from tqdm import tqdm
from datetime import datetime

class MobileNetV3Backbone(torch.nn.Module):
    def __init__(self):
        super(MobileNetV3Backbone, self).__init__()

        self.backbone = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        return torch.flatten(x, 1)
    
class SiamVO(torch.nn.Module):
    def __init__(self):
        super(SiamVO, self).__init__()
        self.backbone = MobileNetV3Backbone()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(576 * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 6)
        )

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        combined = torch.cat((feat1, feat2), dim=1)
        output = self.fc(combined)
        return output