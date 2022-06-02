from sklearn.utils import shuffle
import torch
import torchvision
import torchvision.transforms as transforms

# 加載並標準化CIFAR10
# Torchvision數據集的輸出是[0, 1]範圍的PILImage, 必須將它們進行歸一化到[-1, 1]的Tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # compose疑似是把多個步驟整合在一起，此例就是先轉Tensor再做歸一化

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
