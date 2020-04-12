import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

# for image
import matplotlib.pyplot as plt
import numpy as np

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # self.dataset_name = parameters["dataset"]
        # self.use_cuda = parameters["use_cuda"]

        self.kernel_size = 3
        self.pooling_size = 2

        self.cnn_output_size = 21 ** 2
        self.network_output_size = 208

        self.cnn_layer_2d_patch_scale1 = [
            self.get_network_for_2d_patch_scale_1(), 
            self.get_network_for_2d_patch_scale_1(), 
            self.get_network_for_2d_patch_scale_1()
            ]
        
        self.cnn_layer_2d_patch_scale3 = [
            self.get_network_for_2d_patch_scale_3(), 
            self.get_network_for_2d_patch_scale_3(), 
            self.get_network_for_2d_patch_scale_3()
            ]

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 256), 
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), 
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        # if self.use_cuda:
        #     self.cnn_layer = self.cnn_layer.cuda()
        #     self.fully_connected_layer = self.fully_connected_layer.cuda()
    
    def forward(self, input_value):
        patch_x_scale_1 = input_value["patch_x_scale_1"].float()
        patch_y_scale_1 = input_value["patch_y_scale_1"].float()
        patch_z_scale_1 = input_value["patch_z_scale_1"].float()

        patch_x_scale_3 = input_value["patch_x_scale_3"].float()
        patch_y_scale_3 = input_value["patch_y_scale_3"].float()
        patch_z_scale_3 = input_value["patch_z_scale_3"].float()

        x = self.cnn_layer_2d_patch_scale1[0](patch_x_scale_1) \
            + self.cnn_layer_2d_patch_scale1[1](patch_y_scale_1) \
            + self.cnn_layer_2d_patch_scale1[2](patch_z_scale_1) \
            + self.cnn_layer_2d_patch_scale3[0](patch_x_scale_3) \
            + self.cnn_layer_2d_patch_scale3[1](patch_y_scale_3) \
            + self.cnn_layer_2d_patch_scale3[2](patch_z_scale_3)


        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        return x

    def get_network_for_2d_patch_scale_1(self):
        # outpuesize = 29 - 2 -2 -2 -2 = 21
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3 ,stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
    
    
    def get_network_for_2d_patch_scale_3(self):
        # outpuesize = 87 / 3 -2 -2 -2 = 21
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3 ,stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
    