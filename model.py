import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 250


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(16*128, 512)
        self.fc2 = nn.Linear(512, nclasses)

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn1(self.conv1(x))
        x = F.celu(F.max_pool2d(x, 2))
        x = F.dropout(x, 0.55)
        # print(1, torch.sum(x)/torch.prod(torch.tensor(x.size())))
        x = self.bn2(self.conv2(x))
        x = F.celu(F.max_pool2d(x, 2))
        x = F.dropout(x, 0.55)
        # print(2, torch.sum(x)/torch.prod(torch.tensor(x.size())))
        x = self.bn3(self.conv3(x))
        x = F.celu(F.max_pool2d(x, 2))
        x = F.dropout(x, 0.55)
        # print(3, torch.sum(x)/torch.prod(torch.tensor(x.size())))
        x = x.view(bs, 16*128)
        x = F.celu(self.fc1(x))
        x = F.celu(self.fc2(x))
        return x

class SketchDNN(nn.Module):
    def __init__(self):
        super(SketchDNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=17, stride=3, padding=0)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 4096, kernel_size=7)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        torch.nn.init.xavier_uniform_(self.conv7.weight)
        self.fc1 = nn.Linear(16384, nclasses)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn1(self.conv1(x))
        x = F.max_pool2d(F.relu(x), 3, stride=2, padding=0)
        print(1, torch.sum(x)/torch.prod(torch.tensor(x.size())))
        x = self.bn2(self.conv2(x))
        x = F.max_pool2d(F.relu(x), 2, stride=2, padding=1)
        x = F.relu(self.conv3(x))
        print(2, torch.sum(x)/torch.prod(torch.tensor(x.size())))
        x = F.relu(self.conv4(x))
        print(3, torch.sum(x)/torch.prod(torch.tensor(x.size())))
        x = F.max_pool2d(F.relu(self.conv5(x)), 3, stride=2, padding = 1)
        x = F.dropout(F.relu(self.conv6(x)), 0.55)
        x = F.dropout(F.relu(self.conv7(x)), 0.55)
        print(4, torch.sum(x)/torch.prod(torch.tensor(x.size())))
        x = x.view(bs, -1)
        x = F.relu(self.fc1(x))
        return x
