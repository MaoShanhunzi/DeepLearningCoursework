from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
import torch
from CNNnormalizationNet import CNNNormalizationNet
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "images_original"
train_path = 'train'
val_path = 'val'
test_path = 'test'

transform = transforms.Compose\
    ([transforms.Resize((180,180)),
      transforms.ToTensor(), #[0,255]->[0,1]
      transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #[0,1]->[-1,1]
    ])
train_orig = datasets.ImageFolder(root=os.path.join(path,train_path),transform = transform) #返回map类型 键值对

def data_loader(train_img,batch_size):
    train_loader = DataLoader(train_img, batch_size = batch_size, shuffle=True) #把键值对改为tensor
    return train_loader

def model(train_img, learning_rate=0.001,num_epoch=50, batch_size=32, is_plot=True):
    train_loader = data_loader(train_img,batch_size)
    network = CNNNormalizationNet().to(device)
    optimizer = torch.optim.Adam(network.parameters(),lr=learning_rate)
    cost_func = torch.nn.CrossEntropyLoss()
    costs = []
    m = len(train_img.imgs)
    for epoch in range(num_epoch):
        print("Epoch{}/{}".format(epoch,num_epoch-1))
        print("-"*10)
        loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader): #batch_x为image数据，batch_y为文件夹名字即label
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = network.forward(batch_x)
            cost = cost_func(output,batch_y)
            cost.backward()
            optimizer.step()
            loss += cost.item()
            if step % 16 == 0:
                print(loss)
                print("Batch{}, Train Loss:{:.4f}".format(step+1,loss/(step+1)))
                costs.append(loss/(step+1))

    if is_plot:
        plt.plot(costs)
        plt.xlabel("per 16 steps")
        plt.ylabel('CNNNormalization cost')
        plt.show()

    torch.save(network.state_dict(),'CNNNormalizationNet_para.pkl')
    print('para saved')

def accuracy(dataset,batchsize):
    network = CNNNormalizationNet.to(device)
    network.load_state_dict(torch.load('CNNNormalizationNet_para.pkl'))
    acc_loader = data_loader(dataset,batchsize)
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in acc_loader:
            images, labels = images.to(device),labels.to(device)
            outputs = network.forward(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct +=(predicted == labels).sum().item()
    print('the accuary is {:.4f}'.format(correct/total))

if __name__ == '__main__':
    model(train_orig)
    accuracy(train_orig,32)








