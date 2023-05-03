import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

"""
Imitation learning network
"""
class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=4): 
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=history_length, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.act1 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.2)
        # self.cnn_drop1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2))

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.act2 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.2)
        self.cnn_drop2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.act3 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.2)
        self.cnn_drop3 = nn.Dropout(0.4)
        self.pool3 = nn.MaxPool2d(kernel_size=(2))




        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*3*3, 32)
        self.act_fc1 = nn.LeakyReLU(negative_slope=0.01)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 512)
        self.act_fc2 = nn.LeakyReLU(negative_slope=0.01)
        self.drop2 = nn.Dropout(0.4)
        self.fc_out = nn.Linear(512, n_classes)
        self.output = nn.Softmax(dim=1)
        



    def forward(self, x):
        # TODO: compute forward pass



        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        # x = self.cnn_drop1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.cnn_drop2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.cnn_drop3(x)
        x = self.pool3(x)


        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act_fc2(x)
        x = self.drop2(x)
        x = self.fc_out(x)

        return x