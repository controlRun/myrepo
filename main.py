import os
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
num_epochs = 500
num_classes = 10
batch_size = 64
learning_rate = 0.001

# 数据增广方法
transform_train = transforms.Compose([
    # +4填充至36x36
    transforms.Pad(4),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(), 
    # 随机裁剪至32x32
    transforms.RandomCrop(32), 
    # 转换至Tensor
    transforms.ToTensor(),
    #  归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

transform_test = transforms.Compose([
    # 转换至Tensor
    transforms.ToTensor(),
    #  归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

# cifar10路径
cifarPath = '/home/amax/Jupyter/jxzhu/CIFAR10'

#  训练数据集
train_dataset = torchvision.datasets.CIFAR10(root=cifarPath,
                                             train=True, 
                                             transform=transform_train,
                                             download=False)

# 测试数据集
test_dataset = torchvision.datasets.CIFAR10(root=cifarPath,
                                            train=False, 
                                            transform=transform_test)

# 生成数据加载器
# 训练数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


# In[7]:


model = vgg.VGG_SMALL(10).to(device)
print(model)


max_acc = 0.5#随便设置一个比较小的数，保存精度最高的网络

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    total = 0.0
    
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # 注意模型在GPU中，数据也要搬到GPU中
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * labels.size(0)
        _, pred = torch.max(outputs, 1)     # 预测最大值所在的位置标签
        num_correct = (pred == labels).sum()
        accuracy = (pred == labels).float().mean()
        running_acc += num_correct.item()
        total += labels.size(0)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 200 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.6f} %' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(),100 * running_acc / total))
       
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f} %'.format(
        epoch + 1, running_loss / (len(train_dataset)), 100 * running_acc / (len(train_dataset))))
    
    # 设置为评估模式
    model.eval()

    # 节省计算资源，不去计算梯度
    with torch.no_grad():
        correct = 0
        total = 0
        acc = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        acc = correct / total
        print('Test Accuracy of the model on the test images: {} %'.format(100 * acc))
        
    
    if acc >= max_acc:
        max_acc = acc
        print(max_acc)
        print("save model")
        torch.save(model.state_dict(),'model_dict/vggsmall_b2_c10_dict.pth')

