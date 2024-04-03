import torch
import torch.nn as nn
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=0.4)
        self.relu1 = nn.ReLU()


        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout2d(p=0.4)
        self.relu4 = nn.ReLU()


        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc7 = nn.Linear(in_features=64 * 53 * 53, out_features=128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dropout7 = nn.Dropout(p=0.4)
        self.relu7 = nn.ReLU()

        self.fc8 = nn.Linear(in_features=128, out_features=1)


    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.dropout1(h)
        h = self.relu1(h)

        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool3(h)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.dropout4(h)
        h = self.relu4(h)

        h = self.conv5(h)
        h = self.relu5(h)
        h = self.pool6(h)

        h = h.view(-1, 64 * 53 * 53)

        h = self.fc7(h)
        h = self.bn7(h)
        h = self.dropout7(h)
        h = self.relu7(h)

        y = self.fc8(h)

        return y




class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        # Initial Convolution
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense Block 1 (example with 6 layers, growth rate = 32)
        self.db1_layer1 = self._make_dense_layer(64, 96)
        self.db1_layer2 = self._make_dense_layer(96, 128)
        # Transition Layer 1
        self.trans1_norm = nn.BatchNorm2d(128) #256
        self.trans1_relu = nn.ReLU(inplace=True)
        self.trans1_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)
        self.trans1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Following the same pattern, additional dense blocks and transition layers would be defined here...

        # Final layers
        self.final_norm = nn.BatchNorm2d(64)  # Assuming 128 is the number of features after all blocks
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 1)

    def _make_dense_layer(self, in_channels, growth_rate):
        layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        return layers

    def forward(self, x):
        x = self.pool0(self.relu0(self.norm0(self.conv0(x))))

        # Pass through Dense Block 1
        x = self.db1_layer1(x)
        x = self.db1_layer2(x)

        # Transition Layer 1
        x = self.trans1_norm(x)
        x = self.trans1_relu(x)
        x = self.trans1_conv(x)
        x = self.trans1_pool(x)

        # Additional blocks and transition layers would be processed here...

        # Final processing and classification
        x = self.final_pool(self.final_norm(x))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=1):
        super(Xception, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.entry_flow_conv1 = nn.Sequential(
            SeparableConv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.entry_flow_conv2 = nn.Sequential(
            SeparableConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        x = self.entry_flow_conv1(x)
        x = self.entry_flow_conv2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

