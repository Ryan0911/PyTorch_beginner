from pyexpat import model
import torch
import torchvision
model = torchvision.models.resnet18(pretrained=True)  # 經過預訓練的resnet18模型
data = torch.rand(1, 3, 64, 64)  # 一個隨機tensor，具有3 channels的單張64x64圖像
labels = torch.rand(1, 1000)  # 隨機初始化label，也就是要預測label的值

prediction = model(data)  # forward pass ，正向傳播，通過模型的每一次進行輸入數據用以進行prediction
# 計算loss 通過network反向傳播誤差，調用backword()時就會反向傳播，autograd會為每個model parameter計算梯度並儲存在參數的.grad屬性中
loss = (prediction - labels).sum()
loss.backward()  # backward pass
# optimizer，此例中適用SGD，learning rate=0.01，momentum=0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()  # gradient descent，通過.grad中儲存的梯度調整每個參數
