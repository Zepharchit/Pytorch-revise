#doing from grad calculation using torch only
#import numpy as np 
import torch
# f = w * x

X = torch.tensor([1,2,3,4],dtype=torch.float32)
Y = torch.tensor([2,4,6,8],dtype=torch.float32)

w = torch.tensor(0,dtype=torch.float32,requires_grad=True)
#model preds
def forward(x):
    return w*x

#loss = MSE in LR
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

#grads
#mse = (1/n)*(w*x-y)**x
#dj/dw = (1/n)2(w*x-y)

#def gradient(x,y,y_pred):
 #   return np.dot(2*x,y_pred-y).mean()

print(f'Prediction before:f(5) = {forward(5):.3f}')

#trainiiing
lr = 0.01
n_iter = 20

for epoch in range(n_iter):
    #pred = forward pass
    y_pred = forward(X)
    l = loss(Y,y_pred)
    #backpass
    l.backward()
    
    #update weights
    with torch.no_grad():
        w -= lr * w.grad
    
    
    #zero grads
    w.grad.zero_()

    if epoch % 1 ==0:
        print(f'epoch {epoch+1} : w={w:.3f} : loss={l}')

print(f'Pred after train f(5)={forward(5):.4f}')