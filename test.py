import os
import random
import pathlib
import zipfile
from pathlib import Path
from typing import Tuple, List

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch import nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

image_path= Path('/media/mathews/0C3510680C351068/dev/PYTHON/Machine-Learning/pest_classification/data/test/bees/bees (10).jpg')
img = Image.open(image_path)

data = Path("/media/mathews/0C3510680C351068/dev/PYTHON/Machine-Learning/pest_classification/models/model_resnet_aug_1.pth")
from restnet34 import ResNet34
model = ResNet34(num_classes=12)
model.load_state_dict(torch.load(f=data, weights_only=True))
model.eval()

process = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
    ])

datas = process(img)
with torch.inference_mode():
    pred = model(datas.unsqueeze(0))

print(pred.argmax(dim=1))
