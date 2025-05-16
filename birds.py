import torch.utils
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

data=datasets.CIFAR10(root="./data",train=True,download=False,transform=transforms.ToTensor())
data_val=data=datasets.CIFAR10(root="./data",train=False,download=False,transform=transforms.ToTensor())
data2_val=torch.stack([img for img,_ in data_val],dim=3)
data2=torch.stack([img for img,_ in data],dim=3)
data3=data2.view(3,-1)
data3_val=data2_val.view(3,-1)
data4=datasets.CIFAR10(root="./data",train=True,download=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=data3.mean(dim=1),std=data3.std(dim=1))]))
data4_val=datasets.CIFAR10(root="./data",train=False,download=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=data3_val.mean(dim=1),std=data3_val.std(dim=1))]))
new_data=[(img,label) for img,label in data4 if label==0 or label==2 ]
new_data_val=[(img,label) for img,label in data4_val if label==0 or label==2 ]

if torch.cuda.is_available():
    device="cuda"
    print("cuda")
else:
    device="cpu"
torch.cuda.empty_cache()

model=nn.Sequential(
    nn.Linear(in_features=3072,out_features=1024),
    nn.Tanh(),
    nn.Linear(in_features=1024,out_features=512),
    nn.Tanh(),
    nn.Linear(in_features=512,out_features=128),
    nn.Tanh(),
    nn.Linear(in_features=128,out_features=2)
).to(device=device)

optim= torch.optim.SGD(params=model.parameters(),lr=1e-2)
lossfn=nn.CrossEntropyLoss()
def label_changer(new_data):
    for i in range(0,len(new_data)):
        if new_data[i][1]==2:
            new_data[i]=(new_data[i][0],1)
    return torch.utils.data.DataLoader(dataset=new_data,batch_size=64,shuffle=True)
new_data=label_changer(new_data)
new_data_val=label_changer(new_data_val)
def train(epochs:int):
    for epoch in range(epochs):
        for img,label in new_data:
            batchsize=img.shape[0]
            img=img.view(batchsize,-1).unsqueeze(0)
            img=img.to(device=device)
            out=model(img)
            out=out.squeeze(0)
            loss=lossfn(out,label.to(device=device))
            optim.zero_grad()
            loss.backward()
            optim.step()
        print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
        

train(100)
torch.save(model.state_dict(),"model.pth")
def valuate():
    total=0
    correct=0
    with torch.no_grad():
        for img,labels in new_data_val:
            out=model(img.view(img.shape[0],-1))
            _,predicted=torch.max(out,dim=1)
            total+=labels.shape[0]
            correct+=int((predicted==labels).sum())
    print("Accuracy: %f",correct/total)

valuate()

