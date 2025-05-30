import torch
from matplotlib import pyplot as plt
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c=torch.tensor(t_c)
t_u=torch.tensor(t_u)
def lossfn(t_p,t_u):
    a=(t_p-t_u)**2
    return a.mean()
def model(t_u,w,b):
    return t_u*w+b
t_p=model(t_u,1,2)
loss=lossfn(t_p,t_c)
print(loss)