import torch
import numpy as np
# 與tensor相關的運算操作有很多，包括轉置、索引、切片、數學運算、線代、隨機採樣等等
# 原始文檔超多.. https://pytorch.org/docs/stable/torch.html
# 注意 這些運算都可以在gpu上做使用，相對cpu可以達到更高的運算速度
myTensor = torch.ones(4, 4)
if torch.cuda.is_available():  # 判斷目前環境的GPU是否可用，可用就將tensor移動到gpu做使用
    myTensor = myTensor.to('cuda')
# 1. tensor的索引和切片
print(f"before:\n{myTensor}")
myTensor[:, 1] = 0  # 將第一列的數據全數變0
print(f"after:\n{myTensor}")
print("#######################################\n")
# 2. tensor的拚接 (torch.cat用法，將一組tensor按照指定維度拼接)
t1 = torch.cat([myTensor, myTensor, myTensor], dim=1)
print(f"after cat:\n{t1}")
print("#######################################\n")
# 3. tensor的乘積與矩陣乘法
# 逐個元素相乘
print(f"tensor.mul(tensor): \n {myTensor.mul(myTensor)} \n")
# 等價寫法
print(f"tensor * tensor: \n {myTensor * myTensor}")
# tensor 與 tensor間的矩陣乘法
print(f"tensor.matmul(tensor.T): \n {myTensor.matmul(myTensor.T)} \n")
# 等價寫法
print(f"tensor @ tensor.T: \n {myTensor @ myTensor.T} \n")
print("#######################################\n")

# 4. 自動賦值的運算
# 通常method後都會有_做後綴，操作會直接改變內部取值
print(f"before: \n{myTensor}\n")
myTensor.add_(5)
print(f"after: \n{myTensor}")
# 教材中不鼓勵使用，雖然可節省內存但求導時會因為丟失中間過程而易於導致一些問題
