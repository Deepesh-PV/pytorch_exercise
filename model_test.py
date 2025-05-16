import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
data=datasets.CIFAR10(root="./data",download=False,transform=ToTensor())

for img,label in data:
    if label==0:
        img2=img
        break
model=nn.Sequential(
    nn.Linear(in_features=3072,out_features=512),
    nn.Tanh(),
    nn.Linear(in_features=512,out_features=2),
    nn.Softmax(dim=1)
)
model.load_state_dict(torch.load("model.pth"))
out=model(img2.view(-1).unsqueeze(0))
print(out)