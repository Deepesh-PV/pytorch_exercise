import torch
import torch.nn as nn
import torch.optim as optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c=torch.tensor(t_c).unsqueeze(1)
t_u=torch.tensor(t_u)
m=torch.mean(t_u)
model=nn.Sequential(
    nn.Linear(1,13),
    nn.Tanh(),
    nn.Linear(13,1)
)


loss_fn=nn.L1Loss()

optimizer=optim.Adam(model.parameters(),lr=1e-2)


def trainingloop(epochs,t_u,t_c):
    for i in range(epochs):
        t_p=model(t_u.unsqueeze(1))
        loss=loss_fn(t_c,t_p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%500==0:
            l=(loss/m)*100
            print(f"epoch:{i} loss:{l}%")

trainingloop(epochs=5000,t_u=t_u*0.1,t_c=t_c)
