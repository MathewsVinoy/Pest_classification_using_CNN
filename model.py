import pathlib
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch import nn
from tqdm import tqdm
from restnet34 import ResNet34

MODEL_PATH = Path("models/model_resnet_aug_1.pth")
VALUES = ['ants','bees','beetle','catterpiller','earthworms','earwig','grasshopper','moth','slug','snail','wasp','weevil']

def Model(file_path):
    img = Image.open(file_path)
    model = ResNet34(num_classes=12)
    model.load_state_dict(torch.load(f=MODEL_PATH, weights_only=True))
    model.eval()

    process = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        ])

    data = process(img)
    with torch.inference_mode():
        pred = model(data.unsqueeze(0))
    
    values =pred.argmax(dim=1)
    return VALUES[values] 
    
    

