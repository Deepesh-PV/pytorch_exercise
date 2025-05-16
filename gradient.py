import torch.cuda
import torch
import matplotlib.pyplot as plt

if(torch.cuda.is_available()):
    print("cuda is on")


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c=torch.tensor(t_c)
t_u=torch.tensor(t_u)


def loss_fn(a,b):
    c=(a-b)**2
    return c.mean()

def model_fn(a,w,b): 
    return a*w+b


def dloss(t_p,t_c):
    a=2*(t_p-t_c)/t_p.size(0)
    return a

def dmodel_w(t_u):
    return t_u
def dmodel_b():
    return 1
def gradient_fn(t_u,t_c,w,b):
    gradient_loss=dloss(model_fn(t_u,w,b),t_c)
    weight_gradient=gradient_loss*dmodel_w(t_u)
    bias_gradient=gradient_loss*dmodel_b()
    return torch.stack([weight_gradient.sum(),bias_gradient.sum()])

def training_loop(epoch,t_u,t_c,params,learn):
    for i in range(1,epoch):
        w,b=params
        t_p=model_fn(t_u,w,b)
        loss=loss_fn(t_p,t_c)
        gradient=gradient_fn(t_u,t_c,w,b)
        params=params-learn*gradient
        #print("epoch: %d , loss: %f"%(i,float(loss)))
        #print(f"      params:{params}")
        #print(f"      gradient:{gradient}")
    return params
params=torch.tensor([1.0,0])
t_un=t_u*0.1
params=training_loop(epoch=5000,t_u=t_un,t_c=t_c,params=params,learn=1e-2)
print(params)
t_p=model_fn(t_un,*params)
print(t_p)



fig = plt.figure(dpi=400)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()