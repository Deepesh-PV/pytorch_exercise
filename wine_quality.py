import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split
df=pd.read_csv("wine.csv")
df_out=df['quality']
df_in=df.drop(columns='quality')

t_in=torch.tensor(df_in.values,dtype=torch.float32)
t_out=torch.tensor(df_out.values,dtype=torch.float32)
train_data=t_in[0:int(0.8*len(df))]
val_data=t_in[int(0.8*len(df)):int(0.8*len(df))+int(0.1*len(df))]
test_data=t_in[int(0.8*len(df))+int(0.1*len(df)):len(df)]


train_data_out=t_out[0:int(0.8*len(df))]
val_data_out=t_out[int(0.8*len(df)):int(0.8*len(df))+int(0.1*len(df))]
test_data_out=t_out[int(0.8*len(df))+int(0.1*len(df)):len(df)]
train_mean=torch.mean(train_data_out)
val_mean=torch.mean(val_data_out)
test_mean=torch.mean(test_data_out)

model=nn.Sequential(
    nn.Linear(11,15),
    nn.ReLU(),
    nn.Linear(15,10),
    nn.ReLU(),
    nn.Linear(10,1)
)

optim=torch.optim.SGD(params=model.parameters(),lr=0.2)
loss_fn=nn.L1Loss()

def train(epochs):
    for epoch in range(1,epochs):
        out=model(train_data)
        loss=loss_fn(out.squeeze(1),train_data_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if epoch%100==0:
            print(f"epoch:{epoch} trainoss%={(loss/train_mean)*100}")
            with torch.no_grad():
                out2=model(val_data)
                loss2=loss_fn(out2.squeeze(1),val_data_out)
                print(f"the validationloss:{(loss2/val_mean)*100}")
def test():
    out=model(test_data)
    loss=loss_fn(out.squeeze(1),test_data_out)
    print(f"the test loss:{(loss/test_mean)*100}")
train(500)
test()