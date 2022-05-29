import torch
import numpy as np
# 1. tensor 轉 numpy array
myTensor = torch.ones(5)
print(f"myTensor: {myTensor}")
myNumpy = myTensor.numpy()
print(f"myNumpy: {myNumpy}")
print("#######################################\n")
# 修改tensor的值 numpy array也會跟著改變
myTensor.add_(1)
print(f"myTensor: {myTensor}")
print(f"myNumpy: {myNumpy}")
print("#######################################\n")
# 2. 由numpy array轉 tensor
myNumpy = np.ones(5)
myTensor = torch.from_numpy(myNumpy)
print(f"myTensor: {myTensor}")
print(f"myNumpy: {myNumpy}")
print("#######################################\n")
# 修改numpy array的值 tensor也會跟著改變
np.add(myNumpy, 1, out=myNumpy)
print(f"myTensor: {myTensor}")
print(f"myNumpy: {myNumpy}")
