import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=3),
                         nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
                         nn.Conv1d(out_channels, out_channels, kernel_size=3),
                         nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
                         nn.MaxPool1d(kernel_size=2, stride=2))


class Net(nn.Module):
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        self.conv1 = conv_block(1, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(60 * 512, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)
