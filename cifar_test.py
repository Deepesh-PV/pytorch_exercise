import torch
from torchvision import datasets
from matplotlib import pyplot as plt
data=datasets.CIFAR10("./data")
print(len(data))
img,label=data[99]
plt.imshow(img)
plt.show()
