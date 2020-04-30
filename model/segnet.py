import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

# for image
import matplotlib.pyplot as plt
import numpy as np


class SegNet3DK5L3(nn.Module):

    """
    3D - use 3d image
    K5 - Max kernel size 5
    L3 - CNN leyer 3
    """

    def __init__(self, num_of_class=None, use_cuda=False):
        super(SegNet3DK5L3, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet3DK5L3"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.cnn_output_size = (17 ** 2) * 6 + (4 ** 3)

        self.cnn_layer_2d_patch_scale1 = [
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network()
            ]
        
        self.cnn_layer_2d_patch_scale3 = [
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network()
            ]

        self.cnn_layer_3d_patch = self.get_3d_patch_network()

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(3):
                self.cnn_layer_2d_patch_scale1[i] = self.cnn_layer_2d_patch_scale1[i].cuda()
                self.cnn_layer_2d_patch_scale3[i] = self.cnn_layer_2d_patch_scale3[i].cuda()
            self.cnn_layer_3d_patch = self.cnn_layer_3d_patch.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        if self.use_cuda:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float().cuda()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float().cuda()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float().cuda()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float().cuda()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float().cuda()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float().cuda()

            patch_3d = input_value["patch_3d"].float().cuda()

        else:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float()

            patch_3d = input_value["patch_3d"].float()

        x_1 = self.cnn_layer_2d_patch_scale1[0](patch_x_scale_1)
        x_2 = self.cnn_layer_2d_patch_scale1[1](patch_y_scale_1)
        x_3 = self.cnn_layer_2d_patch_scale1[2](patch_z_scale_1)
        x_4 = self.cnn_layer_2d_patch_scale3[0](patch_x_scale_3)
        x_5 = self.cnn_layer_2d_patch_scale3[1](patch_y_scale_3)
        x_6 = self.cnn_layer_2d_patch_scale3[2](patch_z_scale_3)
        x_7 = self.cnn_layer_3d_patch(patch_3d)

        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_3 = x_3.view(x_3.size(0), -1)
        x_4 = x_4.view(x_4.size(0), -1)
        x_5 = x_5.view(x_5.size(0), -1)
        x_6 = x_6.view(x_6.size(0), -1)
        x_7 = x_7.view(x_7.size(0), -1)

        x = torch.cat((x_1,x_2,x_3,x_4,x_5,x_6,x_7), 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 4 - 4 - 4 ) = 17
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = 87 / 3 - 4 - 4 - 4 =17
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def get_3d_patch_network(self):
        # outpuesize = 13 - 3 - 3 - 3 = 4
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=8, out_channels=4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=4, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


class SegNet3DK5L4(nn.Module):

    """
    3D - use 3d image
    K5 - Max kernel size 5
    L3 - CNN leyer 4
    """

    def __init__(self, num_of_class=None, use_cuda=False):
        super(SegNet3DK5L4, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet3DK5L4"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.cnn_output_size = (13 ** 2) * 6 + (5 ** 3)

        self.cnn_layer_2d_patch_scale1 = [
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network()
            ]
        
        self.cnn_layer_2d_patch_scale3 = [
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network()
            ]

        self.cnn_layer_3d_patch = self.get_3d_patch_network()

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(3):
                self.cnn_layer_2d_patch_scale1[i] = self.cnn_layer_2d_patch_scale1[i].cuda()
                self.cnn_layer_2d_patch_scale3[i] = self.cnn_layer_2d_patch_scale3[i].cuda()
            self.cnn_layer_3d_patch = self.cnn_layer_3d_patch.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        if self.use_cuda:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float().cuda()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float().cuda()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float().cuda()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float().cuda()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float().cuda()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float().cuda()

            patch_3d = input_value["patch_3d"].float().cuda()

        else:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float()

            patch_3d = input_value["patch_3d"].float()

        x_1 = self.cnn_layer_2d_patch_scale1[0](patch_x_scale_1)
        x_2 = self.cnn_layer_2d_patch_scale1[1](patch_y_scale_1)
        x_3 = self.cnn_layer_2d_patch_scale1[2](patch_z_scale_1)
        x_4 = self.cnn_layer_2d_patch_scale3[0](patch_x_scale_3)
        x_5 = self.cnn_layer_2d_patch_scale3[1](patch_y_scale_3)
        x_6 = self.cnn_layer_2d_patch_scale3[2](patch_z_scale_3)
        x_7 = self.cnn_layer_3d_patch(patch_3d)

        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_3 = x_3.view(x_3.size(0), -1)
        x_4 = x_4.view(x_4.size(0), -1)
        x_5 = x_5.view(x_5.size(0), -1)
        x_6 = x_6.view(x_6.size(0), -1)
        x_7 = x_7.view(x_7.size(0), -1)

        x = torch.cat((x_1,x_2,x_3,x_4,x_5,x_6,x_7), 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 4 - 4 - 4 - 4) = 13
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = 87 / 3 - 4 - 4 - 4 -4 =13
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def get_3d_patch_network(self):
        # outpuesize = 13 -2 -2 -2 -2 = 5
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


class SegNet3DK5L5(nn.Module):

    """
    3D - use 3d image
    K5 - Max kernel size 5
    L3 - CNN leyer 5
    """

    def __init__(self, num_of_class=None, use_cuda=False):
        super(SegNet3DK5L5, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet3DK5L5"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.cnn_output_size = (9 ** 2) * 6 + (3 ** 3)

        self.cnn_layer_2d_patch_scale1 = [
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network()
            ]
        
        self.cnn_layer_2d_patch_scale3 = [
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network()
            ]

        self.cnn_layer_3d_patch = self.get_3d_patch_network()

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(3):
                self.cnn_layer_2d_patch_scale1[i] = self.cnn_layer_2d_patch_scale1[i].cuda()
                self.cnn_layer_2d_patch_scale3[i] = self.cnn_layer_2d_patch_scale3[i].cuda()
            self.cnn_layer_3d_patch = self.cnn_layer_3d_patch.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        if self.use_cuda:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float().cuda()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float().cuda()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float().cuda()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float().cuda()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float().cuda()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float().cuda()

            patch_3d = input_value["patch_3d"].float().cuda()

        else:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float()

            patch_3d = input_value["patch_3d"].float()

        x_1 = self.cnn_layer_2d_patch_scale1[0](patch_x_scale_1)
        x_2 = self.cnn_layer_2d_patch_scale1[1](patch_y_scale_1)
        x_3 = self.cnn_layer_2d_patch_scale1[2](patch_z_scale_1)
        x_4 = self.cnn_layer_2d_patch_scale3[0](patch_x_scale_3)
        x_5 = self.cnn_layer_2d_patch_scale3[1](patch_y_scale_3)
        x_6 = self.cnn_layer_2d_patch_scale3[2](patch_z_scale_3)
        x_7 = self.cnn_layer_3d_patch(patch_3d)

        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_3 = x_3.view(x_3.size(0), -1)
        x_4 = x_4.view(x_4.size(0), -1)
        x_5 = x_5.view(x_5.size(0), -1)
        x_6 = x_6.view(x_6.size(0), -1)
        x_7 = x_7.view(x_7.size(0), -1)

        x = torch.cat((x_1,x_2,x_3,x_4,x_5,x_6,x_7), 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 4 - 4 - 4 - 4 - 4) = 9
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = 87 / 3 - 4 - 4 - 4 - 4 - 4 = 9
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def get_3d_patch_network(self):
        # outpuesize = 13 -2 -2 -2 -2 - 2 = 3
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


class SegNet5KP2_ADDALL(nn.Module):

    """
    3D - use 3d image
    K5 - Max kernel size 5
    P2 - Pooling 2
    """

    def __init__(self, use_cuda=False):
        super(SegNet5KP2_ADDALL, self).__init__()
        self.use_cuda = use_cuda

        self.name = "SegNet5KP2_ADDALL"

        self.cnn_output_size = 7 ** 2
        self.network_output_size = 208

        self.cnn_layer_2d_patch_scale1 = [
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network()
            ]
        
        self.cnn_layer_2d_patch_scale3 = [
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network()
            ]

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(3):
                self.cnn_layer_2d_patch_scale1[i] = self.cnn_layer_2d_patch_scale1[i].cuda()
                self.cnn_layer_2d_patch_scale3[i] = self.cnn_layer_2d_patch_scale3[i].cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()
    
    def forward(self, input_value):

        if self.use_cuda:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float().cuda()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float().cuda()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float().cuda()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float().cuda()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float().cuda()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float().cuda()
        else:
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

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 5) / 2 - 5 = 7
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=6 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = ( 87 / 3 - 5) / 2 - 5 = 7
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=6 ,stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


class SegNet5KP2(nn.Module):

    """
    3D - use 3d image
    K5 - Max kernel size 5
    P2 - Pooling 2
    """

    def __init__(self, use_cuda=False):
        super(SegNet5KP2, self).__init__()
        self.use_cuda = use_cuda

        self.name = "SegNet5KP2"

        self.cnn_output_size = (7 ** 2) * 6
        self.network_output_size = 208

        self.cnn_layer_2d_patch_scale1 = [
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network()
            ]
        
        self.cnn_layer_2d_patch_scale3 = [
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network()
            ]

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(3):
                self.cnn_layer_2d_patch_scale1[i] = self.cnn_layer_2d_patch_scale1[i].cuda()
                self.cnn_layer_2d_patch_scale3[i] = self.cnn_layer_2d_patch_scale3[i].cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()
        
        print("MODEL : SegNetKernel5Pool2Concat")
    
    def forward(self, input_value):

        if self.use_cuda:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float().cuda()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float().cuda()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float().cuda()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float().cuda()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float().cuda()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float().cuda()
        else:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float()

        x_1 = self.cnn_layer_2d_patch_scale1[0](patch_x_scale_1)
        x_2 = self.cnn_layer_2d_patch_scale1[1](patch_y_scale_1)
        x_3 = self.cnn_layer_2d_patch_scale1[2](patch_z_scale_1)
        x_4 = self.cnn_layer_2d_patch_scale3[0](patch_x_scale_3)
        x_5 = self.cnn_layer_2d_patch_scale3[1](patch_y_scale_3)
        x_6 = self.cnn_layer_2d_patch_scale3[2](patch_z_scale_3)

        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_3 = x_3.view(x_3.size(0), -1)
        x_4 = x_4.view(x_4.size(0), -1)
        x_5 = x_5.view(x_5.size(0), -1)
        x_6 = x_6.view(x_6.size(0), -1)

        x = torch.cat((x_1,x_2,x_3,x_4,x_5,x_6), 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 5) / 2 - 5 = 7
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=6 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = ( 87 / 3 - 5) / 2 - 5 = 7
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=6 ,stride=1, padding=0),
            nn.ReLU(inplace=True)
        )



class SegNet3D5KP2(nn.Module):

    """
    3D - use 3d image
    K5 - Max kernel size 5
    P2 - Pooling 2
    """

    def __init__(self, use_cuda=False):
        super(SegNetKernel5Pool2Concat3D, self).__init__()
        self.use_cuda = use_cuda

        self.name = "SegNetKernel5Pool2Concat3D"
        self.pooling_size = 2

        self.cnn_output_size = (7 ** 2) * 6 + (7 ** 3)
        self.network_output_size = 208

        self.cnn_layer_2d_patch_scale1 = [
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network()
            ]
        
        self.cnn_layer_2d_patch_scale3 = [
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network()
            ]

        self.cnn_layer_3d_patch = self.get_network_for_3d_patch_scale_1()

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(3):
                self.cnn_layer_2d_patch_scale1[i] = self.cnn_layer_2d_patch_scale1[i].cuda()
                self.cnn_layer_2d_patch_scale3[i] = self.cnn_layer_2d_patch_scale3[i].cuda()
            self.cnn_layer_3d_patch = self.cnn_layer_3d_patch.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        if self.use_cuda:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float().cuda()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float().cuda()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float().cuda()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float().cuda()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float().cuda()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float().cuda()

            patch_3d = input_value["patch_3d"].float().cuda()

        else:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float()

            patch_3d = input_value["patch_3d"].float()

        x_1 = self.cnn_layer_2d_patch_scale1[0](patch_x_scale_1)
        x_2 = self.cnn_layer_2d_patch_scale1[1](patch_y_scale_1)
        x_3 = self.cnn_layer_2d_patch_scale1[2](patch_z_scale_1)
        x_4 = self.cnn_layer_2d_patch_scale3[0](patch_x_scale_3)
        x_5 = self.cnn_layer_2d_patch_scale3[1](patch_y_scale_3)
        x_6 = self.cnn_layer_2d_patch_scale3[2](patch_z_scale_3)
        x_7 = self.cnn_layer_3d_patch(patch_3d)

        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_3 = x_3.view(x_3.size(0), -1)
        x_4 = x_4.view(x_4.size(0), -1)
        x_5 = x_5.view(x_5.size(0), -1)
        x_6 = x_6.view(x_6.size(0), -1)
        x_7 = x_7.view(x_7.size(0), -1)

        x = torch.cat((x_1,x_2,x_3,x_4,x_5,x_6,x_7), 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 5) / 2 - 5 = 7
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=6 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = ( 87 / 3 - 5) / 2 - 5 = 7
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=6 ,stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def get_network_for_3d_patch_scale_1(self):
        # outpuesize = ( 87 / 3 - 5) / 2 - 5 = 7
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),

            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=6 ,stride=1, padding=0),
            nn.ReLU(inplace=True)
        )



class SegNet3DK5L4Cent(nn.Module):

    """
    3D - use 3d image
    K5 - Max kernel size 5
    L3 - CNN leyer 4
    """

    def __init__(self, num_of_class=None, use_cuda=False):
        super(SegNet3DK5L4Cent, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet3DK5L4Cent"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.cnn_output_size = (13 ** 2) * 6 + (5 ** 3) + num_of_class

        self.cnn_layer_2d_patch_scale1 = [
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network()
            ]
        
        self.cnn_layer_2d_patch_scale3 = [
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network()
            ]

        self.cnn_layer_3d_patch = self.get_3d_patch_network()

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(3):
                self.cnn_layer_2d_patch_scale1[i] = self.cnn_layer_2d_patch_scale1[i].cuda()
                self.cnn_layer_2d_patch_scale3[i] = self.cnn_layer_2d_patch_scale3[i].cuda()
            self.cnn_layer_3d_patch = self.cnn_layer_3d_patch.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        if self.use_cuda:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float().cuda()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float().cuda()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float().cuda()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float().cuda()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float().cuda()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float().cuda()

            patch_3d = input_value["patch_3d"].float().cuda()
            
            centroid = input_value["centroid"].float().cuda()

        else:
            patch_x_scale_1 = input_value["patch_x_scale_1"].float()
            patch_y_scale_1 = input_value["patch_y_scale_1"].float()
            patch_z_scale_1 = input_value["patch_z_scale_1"].float()

            patch_x_scale_3 = input_value["patch_x_scale_3"].float()
            patch_y_scale_3 = input_value["patch_y_scale_3"].float()
            patch_z_scale_3 = input_value["patch_z_scale_3"].float()

            patch_3d = input_value["patch_3d"].float()

            centroid = input_value["centroid"].float()

        x_1 = self.cnn_layer_2d_patch_scale1[0](patch_x_scale_1)
        x_2 = self.cnn_layer_2d_patch_scale1[1](patch_y_scale_1)
        x_3 = self.cnn_layer_2d_patch_scale1[2](patch_z_scale_1)
        x_4 = self.cnn_layer_2d_patch_scale3[0](patch_x_scale_3)
        x_5 = self.cnn_layer_2d_patch_scale3[1](patch_y_scale_3)
        x_6 = self.cnn_layer_2d_patch_scale3[2](patch_z_scale_3)
        x_7 = self.cnn_layer_3d_patch(patch_3d)

        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_3 = x_3.view(x_3.size(0), -1)
        x_4 = x_4.view(x_4.size(0), -1)
        x_5 = x_5.view(x_5.size(0), -1)
        x_6 = x_6.view(x_6.size(0), -1)
        x_7 = x_7.view(x_7.size(0), -1)
        centroid = centroid.view(x_7.size(0), -1)

        x = torch.cat((x_1,x_2,x_3,x_4,x_5,x_6,x_7,centroid), 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 4 - 4 - 4 - 4) = 13
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = 87 / 3 - 4 - 4 - 4 -4 =13
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def get_3d_patch_network(self):
        # outpuesize = 13 -2 -2 -2 -2 = 5
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


# class SegNet3DK5L4(nn.Module):

#     """
#     3D - use 3d image
#     K5 - Max kernel size 5
#     L4 - CNN leyer 4
#     """

#     def __init__(self, num_of_class=None, use_cuda=False):
#         super(SegNet3DK5L4, self).__init__()

#         if num_of_class is None:
#             raise ValueError("No no_of_class")

#         self.name = "SegNet3DK5L4"

#         self.use_cuda = use_cuda
#         self.network_output_size = num_of_class

#         self.cnn_output_size = (13 ** 2) * 6 + (4 ** 3)

#         self.cnn_layer_2d_patch_scale1 = [
#             self.get_2d_patch_1_network(), 
#             self.get_2d_patch_1_network(), 
#             self.get_2d_patch_1_network()
#             ]
        
#         self.cnn_layer_2d_patch_scale3 = [
#             self.get_2d_patch_3_network(), 
#             self.get_2d_patch_3_network(), 
#             self.get_2d_patch_3_network()
#             ]

#         self.cnn_layer_3d_patch = self.get_3d_patch_network()

#         self.fully_connected_layer = nn.Sequential(
#             nn.Linear(self.cnn_output_size, 2048),
#             nn.BatchNorm1d(2048), 
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, self.network_output_size),
#             nn.ReLU(inplace=True)
#         )

#         if self.use_cuda:
#             for i in range(3):
#                 self.cnn_layer_2d_patch_scale1[i] = self.cnn_layer_2d_patch_scale1[i].cuda()
#                 self.cnn_layer_2d_patch_scale3[i] = self.cnn_layer_2d_patch_scale3[i].cuda()
#             self.cnn_layer_3d_patch = self.cnn_layer_3d_patch.cuda()
#             self.fully_connected_layer = self.fully_connected_layer.cuda()

#         print("MODEL : {}".format(self.name))
    
#     def forward(self, input_value):

#         if self.use_cuda:
#             patch_x_scale_1 = input_value["patch_x_scale_1"].float().cuda()
#             patch_y_scale_1 = input_value["patch_y_scale_1"].float().cuda()
#             patch_z_scale_1 = input_value["patch_z_scale_1"].float().cuda()

#             patch_x_scale_3 = input_value["patch_x_scale_3"].float().cuda()
#             patch_y_scale_3 = input_value["patch_y_scale_3"].float().cuda()
#             patch_z_scale_3 = input_value["patch_z_scale_3"].float().cuda()

#             patch_3d = input_value["patch_3d"].float().cuda()

#         else:
#             patch_x_scale_1 = input_value["patch_x_scale_1"].float()
#             patch_y_scale_1 = input_value["patch_y_scale_1"].float()
#             patch_z_scale_1 = input_value["patch_z_scale_1"].float()

#             patch_x_scale_3 = input_value["patch_x_scale_3"].float()
#             patch_y_scale_3 = input_value["patch_y_scale_3"].float()
#             patch_z_scale_3 = input_value["patch_z_scale_3"].float()

#             patch_3d = input_value["patch_3d"].float()

#         x_1 = self.cnn_layer_2d_patch_scale1[0](patch_x_scale_1)
#         x_2 = self.cnn_layer_2d_patch_scale1[1](patch_y_scale_1)
#         x_3 = self.cnn_layer_2d_patch_scale1[2](patch_z_scale_1)
#         x_4 = self.cnn_layer_2d_patch_scale3[0](patch_x_scale_3)
#         x_5 = self.cnn_layer_2d_patch_scale3[1](patch_y_scale_3)
#         x_6 = self.cnn_layer_2d_patch_scale3[2](patch_z_scale_3)
#         x_7 = self.cnn_layer_3d_patch(patch_3d)

#         x_1 = x_1.view(x_1.size(0), -1)
#         x_2 = x_2.view(x_2.size(0), -1)
#         x_3 = x_3.view(x_3.size(0), -1)
#         x_4 = x_4.view(x_4.size(0), -1)
#         x_5 = x_5.view(x_5.size(0), -1)
#         x_6 = x_6.view(x_6.size(0), -1)
#         x_7 = x_7.view(x_7.size(0), -1)

#         x = torch.cat((x_1,x_2,x_3,x_4,x_5,x_6,x_7), 1)

#         x = self.fully_connected_layer(x)

#         if self.use_cuda:
#             x = x.cpu()

#         return x

#     def get_2d_patch_1_network(self):
#         # outpuesize = (29 - 4 - 4 - 4 - 4) - 13
#         return nn.Sequential(
#             nn.Conv2d( in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(8),
#             nn.ReLU(inplace=True),

#             nn.Conv2d( in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),

#             nn.Conv2d( in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(8),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5 ,stride=1, padding=0),
#             nn.ReLU(inplace=True),
#         )
    
    
#     def get_2d_patch_3_network(self):
#         # outpuesize = 87 / 3 - 4 - 4 - 4 - 4 =13
#         return nn.Sequential(
#             nn.MaxPool2d(3),

#             nn.Conv2d( in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(8),
#             nn.ReLU(inplace=True),

#             nn.Conv2d( in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),

#             nn.Conv2d( in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(8),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5 ,stride=1, padding=0),
#             nn.ReLU(inplace=True),
#         )

#     def get_3d_patch_network(self):
#         # outpuesize = 13 - 3 - 3 - 3 = 4
#         return nn.Sequential(

#             nn.Conv3d( in_channels=1, out_channels=8, kernel_size=4, stride=1, padding=0),
#             nn.BatchNorm3d(8),
#             nn.ReLU(inplace=True),

#             nn.Conv3d( in_channels=8, out_channels=4, kernel_size=4, stride=1, padding=0),
#             nn.BatchNorm3d(4),
#             nn.ReLU(inplace=True),

#             nn.Conv3d(in_channels=4, out_channels=1, kernel_size=4, stride=1, padding=0),
#             nn.ReLU(inplace=True)
#         )
