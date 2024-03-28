import torch
import torch.nn as nn
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc7 = nn.Linear(in_features=64 * 53 * 53, out_features=128)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=128, out_features=1)
        self.sigmoid8 = nn.Sigmoid()

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool3(h)
        h = self.conv4(h)
        h = self.relu4(h)
        h = self.conv5(h)
        h = self.relu5(h)
        h = self.pool6(h)

        h = h.view(-1, 64 * 53 * 53)

        h = self.fc7(h)
        h = self.relu7(h)
        y = self.fc8(h)
        return y