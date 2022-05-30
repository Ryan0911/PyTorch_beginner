import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
# Q = 3*a^3 - b^2 ，用a和b創造一個tensor Q
Q = 3*a**3 - b**2
# 假設a 和 b是NN的 parameter， Q是誤差
# Q對a的微分是 9*a^2
# Q對b的微分是 -2*b
# 當在Q上使用.backward()，Autograd會計算這些梯度並儲存在各個tensor的.grad attribute中
# gradient是與Q的shape相同的一個tensor
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(f"9*a**2 == a.grad?\n{9*a**2 == a.grad}")
print(f"-2*b  == b.grad?\n{-2*b == b.grad}")
