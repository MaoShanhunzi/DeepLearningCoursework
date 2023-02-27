from torch import nn

in_feature = 180 *180 * 3
hidden_size1 = 200
hidden_size2 = 200
out_feature = 10
class FullyConnectedNet(nn.Module):
    def __init__(self,in_feature,out_feature):
        super().__init__()
        self.layer1 = nn.Linear(in_feature,hidden_size1)
        self.layer2 = nn.Linear(hidden_size1,hidden_size2)
        self.layer3 = nn.Linear(hidden_size2,out_feature)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = x.flatten(start_dim = 1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)

        return x

