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

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCHS = 32

data_path = Path("data_new/new/")
train_dir = data_path / "train"
test_dir = data_path / "test"

data_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir, transform=data_transform)
print(len(train_data))
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)
print(len(test_data))
class_names = train_data.classes
temp=train_data[1]

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCHS, num_workers=0, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCHS, num_workers=0, shuffle=False)

model = ResNet34(num_classes=len(class_names))

EPOCHS = 15
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(EPOCHS)):
    train_loss = 0
    test_loss = 0
    model.train()
    for batch in train_dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = loss_fn(output, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        for batch in test_dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = loss_fn(output, labels)
            test_loss += loss.item()

    print(f"Epoch {epoch}, train loss {train_loss / len(train_dataloader)}, test loss {test_loss / len(test_dataloader)}")

MODEL_PATH = Path("models")
MODEL_NAME = "model_resnet_aug_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


MODEL_PATH.mkdir(parents=True, exist_ok=True)
#Save
torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)
