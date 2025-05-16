from torchvision import transforms,datasets
from matplotlib import pyplot as plt
data=datasets.CIFAR10(root="./data")

img,label=data[99]
transform=transforms.ToTensor()
imgten=transform(img)
plt.imshow(img)
plt.show()
print(imgten.shape)
