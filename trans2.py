import torch
import torchvision
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import math
#batches


class WineData(Dataset):

    def __init__(self,transform=None):
        data = np.loadtxt('wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(data[:,1:])
        self.y = torch.from_numpy(data[:,[0]])
        self.n_samples = data.shape[0]
        self.transform = transform

    def __getitem__(self,index):
        sample =  self.x[index],self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples    

class MulFac:
    def __init__(self,factor):
        self.factor=factor
    def __call__(self,sample):
        inp,lab = sample
        inp *=self.factor
        return inp,lab

composed = torchvision.transforms.Compose([MulFac(2)])
dataset = WineData(transform=composed)

#first_check = dataset[0]
#features,labels = first_check
#print(features,labels)


dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=2)
dataiterator = iter(dataloader)

data = dataiterator.next()
features,labels = data
print(type(features),type(labels))

