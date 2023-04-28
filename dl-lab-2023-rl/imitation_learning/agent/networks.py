import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

"""
Imitation learning network
"""

# class Layer(nn.Module):
#     def __init__(self, in_features, out_features, kernel_size, stride = 1, padding = 0, batch_norm = True):
#         self.in_features = in_features
#         self.out_features = out_features
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.batch_norm = batch_norm

#     def forward(self, x):
#         x = nn.Conv2d(in_channels=self.in_features, out_channels=self.out_features, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
#         if self.batch_norm:
#             x = nn.BatchNorm2d(num_features = self.out_features)
#         return x


class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=4): 
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.block1_conv1 = nn.Conv2d(in_channels=history_length, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block1_bn1 = nn.BatchNorm2d(num_features=32)
        self.block1_act1 = nn.LeakyReLU(negative_slope=0.2)
        self.block1_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block1_bn2 = nn.BatchNorm2d(num_features=64)
        self.block1_act2 = nn.LeakyReLU(negative_slope=0.2)
        self.block1_pool = nn.MaxPool2d(kernel_size=(2))
        
        self.block2_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block2_bn1 = nn.BatchNorm2d(num_features=64)
        self.block2_act1 = nn.LeakyReLU(negative_slope=0.2)
        self.block2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block2_bn2 = nn.BatchNorm2d(num_features=128)
        self.block2_act2 = nn.LeakyReLU(negative_slope=0.2)
        self.block2_pool = nn.AvgPool2d(kernel_size=(24)) #nn.MaxPool2d(kernel_size=(2))

        self.block3_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.block3_bn1 = nn.BatchNorm2d(num_features=128)
        self.block3_act1 = nn.LeakyReLU(negative_slope=0.2)
        self.block3_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.block3_bn2 = nn.BatchNorm2d(num_features=128)
        self.block3_act2 = nn.LeakyReLU(negative_slope=0.2)
        self.block3_pool = nn.AvgPool2d(kernel_size=(12))

        self.conv_layers = [
            self.block1_conv1, self.block1_bn1, self.block1_act1, self.block1_conv2, self.block1_bn2, self.block1_act2, self.block1_pool,
            self.block2_conv1, self.block2_bn1, self.block2_act1, self.block2_conv2, self.block2_bn2, self.block2_act2, self.block2_pool,
            # self.block3_conv1, self.block3_bn1, self.block3_act1, self.block3_conv2, self.block3_bn2, self.block3_act2, self.block3_pool, 
        ]

        # self.conv1 = nn.Conv2d(in_channels=history_length, out_channels=32, kernel_size=3, stride=1, padding='same')
        # self.bn1 = nn.BatchNorm2d(num_features=32)
        # self.act1 = nn.LeakyReLU(negative_slope=0.2)
        # self.pool1 = nn.MaxPool2d(kernel_size=(2))
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        # self.bn2 = nn.BatchNorm2d(num_features=64)
        # self.act2 = nn.LeakyReLU(negative_slope=0.2)
        # self.pool2 = nn.AvgPool2d(kernel_size=(48))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 32)
        self.act_fc1 = nn.LeakyReLU(negative_slope=0.2)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 256)
        self.act_fc2 = nn.LeakyReLU(negative_slope=0.2)
        self.drop2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, n_classes)
        self.output = nn.Softmax(dim=1)
        



    def forward(self, x):
        # TODO: compute forward pass
        inp = x
        count = 0
        for layer in self.conv_layers:
            # if count!= 1 and count!= 4 and count!= 8 and count!= 11:
            x = layer(x)
            count += 1
            if count == 1 or count == 8: # or count == 15:
                inp = x
            if count == 4 or count == 11: # or count == 18:
               x = torch.cat((x, inp), dim=1)



        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.act1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.act2(x)
        # x = self.pool2(x)


        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act_fc2(x)
        x = self.drop2(x)
        x = self.fc_out(x)
        # x = self.output(x)

        return x