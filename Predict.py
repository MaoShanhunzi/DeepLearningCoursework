import torch
from FullyConnectedNet import FullyConnectedNet
from CNNNet import CNNNet
from CNNnormalizationNet import CNNNormalizationNet
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from SplitData import joint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose\
    ([transforms.Resize((180,180)),
      transforms.ToTensor(),
      transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
      ])
path = 'images_original'
predict_path = 'test'
test_path = joint(path,predict_path)
test_orig = datasets.ImageFolder(root=os.path.join(path,predict_path),transform=transform)

Network = 'FullyConnectedNet'


def data_loader(train_img,batch_size):
    train_loader = DataLoader(train_img, batch_size=batch_size,shuffle=True)
    return train_loader

def accuracy(dataset,batchsize,Choose_Network):
    if Choose_Network == 'FullyConnectedNet':
        network = FullyConnectedNet(180 * 180 * 3).to(device)
        network.load_state_dict(torch.load('FullyConnectedNet_para.pkl'))
        print('FullyConnectedNet load sucessfully')
    elif Choose_Network == 'CNNNet':
        network = CNNNet().to(device)
        network.load_state_dict(torch.load('CNNNet_para.pkl'))
        print('FullyConnectNet load sucessfully')
    elif Choose_Network == 'CNNNormalizationNet':
        network = CNNNormalizationNet().to(device)
        network.load_state_dict(torch.load('CNNNormalizationNet_para.pkl'))
        print('CNNNormalizationNet load sucessfully')
    else:
        print('还有一个参数加载成功')

    acc_loader=data_loader(dataset,batchsize)
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
        for images, labels in acc_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = network.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            print("{}的预测值为{}".format(labels,predicted))
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # 将预测及标签两相同大小张量逐一比较各相同元素的个数
            #.item()将tensor类别的int值转成python数字
    print('the accuracy is {:.4f}'.format(correct / total))

if __name__ == '__main__':
    accuracy(test_orig,64,Network)