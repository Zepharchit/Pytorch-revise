import numpy as np 
import torch

x = torch.rand(3,requires_grad=True)
print(x)

y = x*x+2#FP
print(y)

y.backward(torch.FloatTensor([1.0, 1.0, 1.0]))#dy/dx gradients
#need to add torch.floatTensor - kind of weight else throws errors
#tensor provided to .backward should be 1 of same dimension as input since it gets multiplied by The jacobian mat/partial derivatives



print(x.grad)

# to remove grads
#x.requires_grad_()
#x.detach()
#with torch.no_grad():

#x.requires_grad_(False)# trailing underscore function, changes made will be inplace

#z = x.detach()

with torch.no_grad():
    z = x+2

print(z)


weights = torch.ones(4,requires_grad=True)
for epoch in range(3):
    m_o = (weights*3).sum()
    m_o.backward()
    print(weights.grad)
    # the changes to weights gets added up hence we need to make grads zero with each iteration

    weights.grad.zero_()

optimizer = torch.optim.SGD(weights,lr=0.1)
optimizer.step()
optimizer.zero_grad()
#the changes to gradients must be reset after each iteration