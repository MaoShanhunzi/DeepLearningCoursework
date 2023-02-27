from torch import nn

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        #input is (3*180*180)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3)
        self.relu = nn.ReLU()
        #input is (16*178*178)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3)
        #input is (16*176*176)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        #input is (16*88*88)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        #input is (32*86*86)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3)
        # input is (32*84*84)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        #input is (32*42*42)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*42*42,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x