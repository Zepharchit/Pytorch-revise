import torch
import torchvision
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import math
#batches


class WineData(Dataset):

    def __init__(self):
        data = np.loadtxt('wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(data[:,1:])
        self.y = torch.from_numpy(data[:,[0]])
        self.n_samples = data.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.n_samples    


dataset = WineData()

#first_check = dataset[0]
#features,labels = first_check
#print(features,labels)


dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=2)
dataiterator = iter(dataloader)

data = dataiterator.next()
features,labels = data
print(features,labels)


# training loop
epochs = 5
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples,n_iterations)



for epoch in range(epochs):
    for i,(ip,lab) in enumerate(dataloader):
        if (i+1)%5 ==0:
            print(f'epochs :{epoch+1}/{epochs} step:{i+1}/{n_iterations}')