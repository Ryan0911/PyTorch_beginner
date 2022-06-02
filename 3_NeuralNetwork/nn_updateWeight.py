import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import LeNet
# modle creation
net = LeNet.Net()
# create input & output & dummy target
input = torch.randn(1, 1, 32, 32)
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
# compute loss
criterion = nn.MSELoss()
loss = criterion(output, target)
net.zero_grad()
loss.backward()
# update weight
learning_rate = 0.01
for f in net._parameters():
    f.data.sub_(f.grad.data * learning_rate)
# torch.optim有各種不同的參數更新規則，例如: SGD、Nesterov-SGD、Adam、RMSProp等
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
# in your traning loop:
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # does the update
