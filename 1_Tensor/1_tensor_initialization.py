import torch
import numpy as np
# Tensor initialization, 4 examples
# 1. 直接生成
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"x_data: \n {x_data} \n")  # print前面加f表示格式化字串 相當於.format的用法
print("#######################################\n")
# 2. Numpy生成Tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)  # 也可以從tensor轉回numpy
print(f"np_array: \n {np_array} \n")
print(f"x_np: \n {x_np} \n")
print("#######################################\n")
# 3. 用以有的tensor生成新的tensor
x_ones = torch.ones_like(x_data)  # 目的是創建一個和目標參數維度一樣且元素都為1的tensor
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # int -> float
print(f"Random Tensor: \n {x_rand} \n")
print("#######################################\n")
# 4. 通過指定數據維度生成tensor
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} ")
