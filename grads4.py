#doing from grad calculation using torch only
# Training Pipeline
# 1. Designnn model (input,output size, FP)
# 2. Construct loss and optimizer
# 3. Training Loop
#   -FP;- compute pred
#   -BP;- grads
#   -update weights

import torch
# f = w * x
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

n_sample,n_feature = X.shape
print(n_sample,n_feature)

X_test = torch.tensor([5],dtype=torch.float32)
input_size = n_feature
output_size = n_feature
#w = torch.tensor(0,dtype=torch.float32,requires_grad=True)
#model preds
#def forward(x):
    #return w*x

# not defining loss manually loss = MSE in LR
#def loss(y,y_pred):
    #return ((y_pred-y)**2).mean()

#grads
#mse = (1/n)*(w*x-y)**x
#dj/dw = (1/n)2(w*x-y)

#def gradient(x,y,y_pred):
 #   return np.dot(2*x,y_pred-y).mean()

model = nn.Linear(input_size,output_size)

print(f'Prediction before:f(5) = {model(X_test).item():.3f}')

#trainiiing
lr = 0.01
n_iter = 35
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr)


for epoch in range(n_iter):
    #pred = forward pass
    y_pred = model(X)
    l = loss(Y,y_pred)
    #backpass
    l.backward()
    
    #update weights
    #with torch.no_grad():
     #   w -= lr * w.grad
    optimizer.step() #for updating weights
    
    #zero grads
    #w.grad.zero_()
    optimizer.zero_grad()
    if epoch % 1 ==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1} : w={w[0][0].item():.3f} : loss={l}')

print(f'Pred after train f(5)={model(X_test).item():.4f}')