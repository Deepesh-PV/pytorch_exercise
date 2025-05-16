from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
import torch
data= datasets.CIFAR10(root="./data",transform=transforms.ToTensor(),train=True)
imgs=torch.stack([img for img,_ in data],dim=3)
print(imgs.shape)
imgs_n=imgs.view(3,-1)
data2=datasets.CIFAR10(root="./data",transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=imgs_n.mean(dim=1),std=imgs_n.std(dim=1))]),train=True)
plt.imshow(data2[99][0].permute(1,2,0))
plt.show()