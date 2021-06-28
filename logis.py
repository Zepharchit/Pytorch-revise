import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = datasets.load_breast_cancer()

X,y = df.data,df.target
n_sample,n_feature=X.shape

print(n_sample)
print(n_feature)
print(X)

sc = StandardScaler()
X = sc.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)
print(X_train.shape)
print(y_train.shape)

#convert to torch tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)



#model
class Logreg(nn.Module):
    def __init__(self,input_features):
        super(Logreg, self).__init__()
        self.lin = nn.Linear(input_features,1)
        
    def forward(self,x):
        y_pred = torch.sigmoid(self.lin(x))
        return y_pred
    

model = Logreg(n_feature)
print(model)
criterion = nn.BCELoss()
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=lr)

epochs = 50
for epoch in range(epochs):
    #Fp
    y_pred = model(X_train)
    loss = criterion(y_pred,y_train)

    #Bp
    loss.backward()
    #updates
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)%2==0:
        print(f'epoch:{epoch+1},loss:{loss.item():.4f}')


with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cl = y_pred.round()
    acc = y_pred_cl.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy {acc:.4f}')