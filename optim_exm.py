import torch
import torch.optim as optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c=torch.tensor(t_c)
t_u=torch.tensor(t_u)

param=torch.tensor([1.0,0.0],requires_grad=True)


optimizer=optim.Adam([param],lr=1e-2)
def lossfn(t_p,t_c):
    a=(t_p-t_c)**2
    return a.mean()

def model(t_u,w,b):
    return w*t_u+b


def trainingloop(epochs,params,t_u,t_c,learn):
    for i in range(epochs):
        w,b= params
        t_p=model(t_u=t_u,w=w,b=b)
        loss=lossfn(t_c=t_c,t_p=t_p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%500==0:
            print("epoch:%d loss:%f"%(i,loss))

trainingloop(epochs=5000,params=param,t_u=t_u*0.1,t_c=t_c,learn=1e-2)

