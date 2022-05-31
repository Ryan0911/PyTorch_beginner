import imp
from torch import nn, optim
import torchvision
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
# 假設我們要在具有10個標籤的新資料集中進行微調。
# 在resnet中，分類器是最後一個線性層model.fc。
#我們可以簡單替換為充當我們分類器的新線性層(默認情況下requires_grad = True(未凍結))
model.fc = nn.Linear(512, 10)

# 現在，除了model.fc的參數外，模型中所有參數都將凍結。
# 計算梯度的唯一參數為model.fc的權重和偏差
# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
# 雖在優化器中註冊所有參數，但唯一可計算梯度的參數，是分類器的權重與偏差。
# torch.no_grad()中的上下文管理器可以使用相同的排除功能。
