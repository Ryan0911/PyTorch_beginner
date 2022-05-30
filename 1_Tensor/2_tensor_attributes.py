import torch
import numpy as np
# 學學如何得到tensor的維度、數據類型與儲存的設備(GPU or CPU)
myTensor = torch.rand(3, 4)

print(f"Shape of tensor: {myTensor.shape}")
print(f"Datatype of tensor: {myTensor.dtype}")
print(f"Device tensor is stored on: {myTensor.device}")
# 若移動到gpu上
myTensor = myTensor.to(device='cuda')
print(f"Revised device tensor is stored on: {myTensor.device}")
