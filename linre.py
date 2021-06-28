import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt 


# data preparation
X_numpy, y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

#print(X_numpy.shape)
#print(y_numpy.shape)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0],1)# used to change contiguous data

n_sample,n_feature = X.shape
input_size = n_feature
output_size = 1
#print(n_feature)
#print(n_sample)
model = nn.Linear(input_size,output_size)

#loss
criterion = nn.MSELoss()
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
epochs = 50

for epoch in range(epochs):

    #Fp
    y_pred = model(X)
    loss = criterion(y_pred,y)

    #BP9
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 2 ==0:
        print(f'epoch:{epoch+1},loss:{loss.item():.3f}')


pre = model(X).detach().numpy()# to remove require grads = TRue
plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,pre,'b')

plt.show()