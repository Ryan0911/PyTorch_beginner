import torch
import torch.nn as nn
import torch.nn.functional as F
import LeNet

net = LeNet.Net()
print(net)
print("-----------------------------------------\n")
# 模型的可學習參數可由net.parameters() return.
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
# print(params)
print("-----------------------------------------\n")
# 嘗試一個 32x32的隨機輸入
# 注意: 該網路的預期輸入大小(LeNet)為 32x32
# 若要使用MNIST dataset，要將圖像調整為32x32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# 使用隨機梯度將所有參數和backward pass的梯度緩衝區歸零
net.zero_grad()
out.backward(torch.rand(1, 10))
# torch.nn只支持小批量，只支持微型樣本而非單樣本的輸入
# ex: nn.Conv2d採用 nSample x nChannels x Height x Width 的 4D Tensor
# 若只有一個樣本，必須使用input.unsqueeze(0) 添加一個"假"批量，以符合套件的要求
