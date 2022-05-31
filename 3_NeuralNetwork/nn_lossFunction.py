import torch
import torch.nn as nn
import torch.nn.functional as F
import LeNet
net = LeNet.Net()
input = torch.randn(1, 1, 32, 32)
output = net(input)
target = torch.randn(10)  # a dummy(虛擬) target, for example.
print(f"Before view: \n{target.shape}")
target = target.view(1, -1)  # make it the same shape as output
# 在PyTorch中，view的作用是重構tensor的維度，相當於numpy的resize()
print(f"After view: \n{target.shape}")
print("------------------------------------------------------")
criterion = nn.MSELoss()
loss = criterion(output, target)
print(f"Loss: \n{loss}")
# 使用.grad_fn 屬性向後追蹤loss，會看到一個計算圖
# 調用loss.backward()時，會對整個圖微分。
# 圖中具有requires_grad = True 的所有Tensor將隨梯度累積至.grad tensor.
# 就有點像...下面的東西，其實有點抽象的感覺
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
print("------------------------------------------------------")
# backward pass
# 此時只需要做loss.backward() 但必須要清除現有的梯度，否則梯度會累積至現有的梯度中
net.zero_grad()  # zeros the gradient buffers of all parameters
print('conv1.bias.grad before backward: ')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward: ')
print(net.conv1.bias.grad)
