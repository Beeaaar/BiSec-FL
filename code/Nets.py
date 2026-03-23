import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dims=(512, 128),
        num_classes=10,
        dropout=0.3,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(self.dropout(x))
        return F.log_softmax(x, dim=1)

class LeNetBN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        conv_channels=(32, 64),
        fc_dim=2048
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(1, conv_channels[0], kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])

        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])

        self.fc1 = nn.Linear(7 * 7 * conv_channels[1], fc_dim)
        self.bn3 = nn.BatchNorm1d(fc_dim)

        self.fc2 = nn.Linear(fc_dim, num_classes)
        self.bn4 = nn.BatchNorm1d(num_classes)

        self.act = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)

        x = self.act(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = self.act(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = self.act(self.bn3(self.fc1(x)))
        x = self.bn4(self.fc2(x))

        return F.log_softmax(x, dim=1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
