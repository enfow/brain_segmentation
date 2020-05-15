import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

# for image
import matplotlib.pyplot as plt
import numpy as np


# SegNet3DK5L4IdentCentLargeFC
class SegNet_CNN4A_FC6C(nn.Module):

    """
    CNN4A - CNN 4 layer with size A
    FC6C - FC 6 layer with size C
    """

    def __init__(
                self, 
                num_of_class=None, 
                use_centroid=True,
                use_cuda=True,
                noise_size=None
                ):
        super(SegNet_CNN4A_FC6C, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet_CNN4A_FC6C"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.use_centroid = use_centroid
        self.noise_size = noise_size

        self.cnn_output_size = (13 ** 2) * 6 + (5 ** 3)
        if self.use_centroid:
            self.cnn_output_size += num_of_class

        self.cnn_layers =[
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(),
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(),
            self.get_3d_patch_network(),
        ]

        self.centroid_identity_layer = nn.Sequential(
            nn.Linear(num_of_class, num_of_class),
            nn.BatchNorm1d(num_of_class), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_of_class, num_of_class),
            nn.ReLU(inplace=True),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 4096),
            nn.BatchNorm1d(4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(len(self.cnn_layers)):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()
            self.centroid_identity_layer = self.centroid_identity_layer.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        cnn_input_list = [
            input_value["patch_x_scale_1"].float(),
            input_value["patch_y_scale_1"].float(),
            input_value["patch_z_scale_1"].float(),

            input_value["patch_x_scale_3"].float(),
            input_value["patch_y_scale_3"].float(),
            input_value["patch_z_scale_3"].float(),

            input_value["patch_3d"].float()
        ]

        cnn_outputs=[]
        if self.use_cuda:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data.cuda())
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)
        else:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data)
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)

        if self.use_centroid:
            if self.use_cuda:
                centroid = input_value["centroid"].float().cuda()
            else:
                centroid = input_value["centroid"].float()
            if isinstance(self.noise_size, float):
                noise_of_centroid = (torch.randn_like(centroid) * self.noise_size)
                centroid = (centroid + noise_of_centroid).clamp(0,1)

            centroid_output = self.centroid_identity_layer(centroid)
            cnn_outputs.append(centroid_output.view(centroid_output.size(0), -1))

        x = torch.cat(cnn_outputs, 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # output size = (29 - 4 - 4 - 4 - 4) = 13
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
        # output size = 87 / 3 - 4 - 4 - 4 -4 =13
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
        # output size = 13 -2 -2 -2 -2 = 5
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


class SegNet_CNN2A_FC6C(nn.Module):

    """
    CNN2A - CNN 2 layer with size A
    FC6C - FC 6 layer with size C
    """

    def __init__(
                self, 
                num_of_class=None, 
                use_centroid=True,
                use_cuda=True,
                noise_size=None
                ):
        super(SegNet_CNN2A_FC6C, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet_CNN2A_FC6C"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.use_centroid = use_centroid
        self.noise_size = noise_size

        self.cnn_output_size = (21 ** 2) * 6 + (9 ** 3)
        if self.use_centroid:
            self.cnn_output_size += num_of_class

        self.cnn_layers =[
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(),
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(),
            self.get_3d_patch_network(),
        ]

        self.centroid_identity_layer = nn.Sequential(
            nn.Linear(num_of_class, num_of_class),
            nn.BatchNorm1d(num_of_class), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_of_class, num_of_class),
            nn.ReLU(inplace=True),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 4096),
            nn.BatchNorm1d(4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(len(self.cnn_layers)):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()
            self.centroid_identity_layer = self.centroid_identity_layer.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        cnn_input_list = [
            input_value["patch_x_scale_1"].float(),
            input_value["patch_y_scale_1"].float(),
            input_value["patch_z_scale_1"].float(),

            input_value["patch_x_scale_3"].float(),
            input_value["patch_y_scale_3"].float(),
            input_value["patch_z_scale_3"].float(),

            input_value["patch_3d"].float()
        ]

        cnn_outputs=[]
        if self.use_cuda:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data.cuda())
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)
        else:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data)
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)

        if self.use_centroid:
            if self.use_cuda:
                centroid = input_value["centroid"].float().cuda()
            else:
                centroid = input_value["centroid"].float()
            if isinstance(self.noise_size, float):
                noise_of_centroid = (torch.randn_like(centroid) * self.noise_size)
                centroid = (centroid + noise_of_centroid).clamp(0,1)

            centroid_output = self.centroid_identity_layer(centroid)
            cnn_outputs.append(centroid_output.view(centroid_output.size(0), -1))

        x = torch.cat(cnn_outputs, 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 4 - 4 ) = 21
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = 87 / 3 - 4 - 4 = 21
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),

        )

    def get_3d_patch_network(self):
        # outpuesize = 13 -2 -2 = 9
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

        )


class SegNet_CNN3C_FC6C(nn.Module):

    """
    CNN3C - CNN 3 layer with size C
    FC6C - FC 6 layer with size C
    """

    def __init__(
                self, 
                num_of_class=None, 
                use_centroid=True,
                use_cuda=True,
                noise_size=None
                ):
        super(SegNet_CNN3C_FC6C, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet_CNN3C_FC6C"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.use_centroid = use_centroid
        self.noise_size = noise_size

        self.cnn_output_size = (17 ** 2) * 6 + (4 ** 3)
        if self.use_centroid:
            self.cnn_output_size += num_of_class

        self.cnn_layers =[
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(),
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(),
            self.get_3d_patch_network(),
        ]

        self.centroid_identity_layer = nn.Sequential(
            nn.Linear(num_of_class, num_of_class),
            nn.BatchNorm1d(num_of_class), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_of_class, num_of_class),
            nn.ReLU(inplace=True),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 4096),
            nn.BatchNorm1d(4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(len(self.cnn_layers)):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()
            self.centroid_identity_layer = self.centroid_identity_layer.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        cnn_input_list = [
            input_value["patch_x_scale_1"].float(),
            input_value["patch_y_scale_1"].float(),
            input_value["patch_z_scale_1"].float(),

            input_value["patch_x_scale_3"].float(),
            input_value["patch_y_scale_3"].float(),
            input_value["patch_z_scale_3"].float(),

            input_value["patch_3d"].float()
        ]

        cnn_outputs=[]
        if self.use_cuda:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data.cuda())
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)
        else:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data)
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)

        if self.use_centroid:
            if self.use_cuda:
                centroid = input_value["centroid"].float().cuda()
            else:
                centroid = input_value["centroid"].float()
            if isinstance(self.noise_size, float):
                noise_of_centroid = (torch.randn_like(centroid) * self.noise_size)
                centroid = (centroid + noise_of_centroid).clamp(0,1)

            centroid_output = self.centroid_identity_layer(centroid)
            cnn_outputs.append(centroid_output.view(centroid_output.size(0), -1))

        x = torch.cat(cnn_outputs, 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 4 - 4 - 4 ) = 17
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = 87 / 3 - 4 - 4 - 4 =17
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def get_3d_patch_network(self):
        # outpuesize = 13 - 3 - 3 - 3 = 4
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=32, out_channels=8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=8, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


class SegNet_CNN4D_FC5C(nn.Module):

    """
    CNN4D - CNN 4 layer with size D
    FC5C - FC 5 layer with size C
    """

    def __init__(
                self, 
                num_of_class=None, 
                use_centroid=True,
                use_cuda=True,
                noise_size=None
                ):
        super(SegNet_CNN4D_FC5C, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet_CNN4D_FC5C"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.use_centroid = use_centroid
        self.noise_size = noise_size

        self.cnn_output_size = (13 ** 2) * 6 + (7 ** 3)
        if self.use_centroid:
            self.cnn_output_size += num_of_class

        self.cnn_layers =[
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(),
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(),
            self.get_3d_patch_network(),
        ]

        self.centroid_identity_layer = nn.Sequential(
            nn.Linear(num_of_class, num_of_class),
            nn.BatchNorm1d(num_of_class), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_of_class, num_of_class),
            nn.ReLU(inplace=True),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 4096),
            nn.BatchNorm1d(4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.network_output_size),
            nn.ReLU(inplace=True)
        )

        if self.use_cuda:
            for i in range(len(self.cnn_layers)):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()
            self.centroid_identity_layer = self.centroid_identity_layer.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        cnn_input_list = [
            input_value["patch_x_scale_1"].float(),
            input_value["patch_y_scale_1"].float(),
            input_value["patch_z_scale_1"].float(),

            input_value["patch_x_scale_3"].float(),
            input_value["patch_y_scale_3"].float(),
            input_value["patch_z_scale_3"].float(),

            input_value["patch_3d"].float()
        ]

        cnn_outputs=[]
        if self.use_cuda:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data.cuda())
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)
        else:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data)
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)

        if self.use_centroid:
            if self.use_cuda:
                centroid = input_value["centroid"].float().cuda()
            else:
                centroid = input_value["centroid"].float()
            if isinstance(self.noise_size, float):
                noise_of_centroid = (torch.randn_like(centroid) * self.noise_size)
                centroid = (centroid + noise_of_centroid).clamp(0,1)

            centroid_output = self.centroid_identity_layer(centroid)
            cnn_outputs.append(centroid_output.view(centroid_output.size(0), -1))

        x = torch.cat(cnn_outputs, 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 4 - 4 - 4 - 4 ) = 13
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=32, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = 87 / 3 - 4 - 4 - 4 - 4 =13
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=32, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5 ,stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def get_3d_patch_network(self):
        # outpuesize = 13 - 2 - 2 - 2 = 7
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


class SegNet_CNN3D_FC7C1(nn.Module):

    """
    CNN3D - CNN 3 layer with size D
    FC7C - FC 7 layer with size C1
    """

    def __init__(
                self, 
                num_of_class=None, 
                use_centroid=True,
                use_cuda=True,
                noise_size=None
                ):
        super(SegNet_CNN3D_FC7C1, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet_CNN3D_FC7C1"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.use_centroid = use_centroid
        self.noise_size = noise_size

        self.cnn_output_size = (17 ** 2) * 6 + (7 ** 3)
        if self.use_centroid:
            self.cnn_output_size += num_of_class

        self.cnn_layers =[
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(), 
            self.get_2d_patch_1_network(),
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(), 
            self.get_2d_patch_3_network(),
            self.get_3d_patch_network(),
        ]

        self.centroid_identity_layer = nn.Sequential(
            nn.Linear(num_of_class, num_of_class),
            nn.BatchNorm1d(num_of_class), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_of_class, num_of_class),
            nn.ReLU(inplace=True),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096), 
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.network_output_size),
            nn.ReLU(inplace=True)
            ) 

        if self.use_cuda:
            for i in range(len(self.cnn_layers)):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()
            self.centroid_identity_layer = self.centroid_identity_layer.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        cnn_input_list = [
            input_value["patch_x_scale_1"].float(),
            input_value["patch_y_scale_1"].float(),
            input_value["patch_z_scale_1"].float(),

            input_value["patch_x_scale_3"].float(),
            input_value["patch_y_scale_3"].float(),
            input_value["patch_z_scale_3"].float(),

            input_value["patch_3d"].float()
        ]

        cnn_outputs=[]
        if self.use_cuda:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data.cuda())
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)
        else:
            for idx, data in enumerate(cnn_input_list):
                output = self.cnn_layers[idx](data)
                output = output.view(output.size(0), -1)
                cnn_outputs.append(output)

        if self.use_centroid:
            if self.use_cuda:
                centroid = input_value["centroid"].float().cuda()
            else:
                centroid = input_value["centroid"].float()
            if isinstance(self.noise_size, float):
                noise_of_centroid = (torch.randn_like(centroid) * self.noise_size)
                centroid = (centroid + noise_of_centroid).clamp(0,1)

            centroid_output = self.centroid_identity_layer(centroid)
            cnn_outputs.append(centroid_output.view(centroid_output.size(0), -1))

        x = torch.cat(cnn_outputs, 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_1_network(self):
        # outpuesize = (29 - 4 - 4 - 4 ) = 17
        return nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    
    def get_2d_patch_3_network(self):
        # outpuesize = 87 / 3 - 4 - 4 - 4 = 17
        return nn.Sequential(
            nn.MaxPool2d(3),

            nn.Conv2d( in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def get_3d_patch_network(self):
        # outpuesize = 13 - 2 - 2 - 2 = 7
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


class SegNet_CNN21D_FC7C1(nn.Module):

    """
    CNN21D - CNN 2 sharing layer + 1 layer with size D
    FC7C - FC 7 layer with size C1
    """

    def __init__(
                self, 
                num_of_class=None, 
                use_centroid=True,
                use_cuda=True,
                noise_size=None
                ):
        super(SegNet_CNN21D_FC7C1, self).__init__()

        if num_of_class is None:
            raise ValueError("No no_of_class")

        self.name = "SegNet_CNN21D_FC7C1"

        self.use_cuda = use_cuda
        self.network_output_size = num_of_class

        self.use_centroid = use_centroid
        self.noise_size = noise_size

        self.cnn_output_size = (17 ** 2) * 6 + (7 ** 3)
        if self.use_centroid:
            self.cnn_output_size += num_of_class

        self.cnn_sharing_layer_patch_1_network = nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )

        self.cnn_sharing_layer_patch_3_network = nn.Sequential(
            nn.MaxPool2d(3),
            nn.Conv2d( in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d( in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )

        self.cnn_layers =[
            self.get_2d_patch_network(), 
            self.get_2d_patch_network(), 
            self.get_2d_patch_network(),
            self.get_2d_patch_network(), 
            self.get_2d_patch_network(), 
            self.get_2d_patch_network(),
            self.get_3d_patch_network(),
        ]

        self.centroid_identity_layer = nn.Sequential(
            nn.Linear(num_of_class, num_of_class),
            nn.BatchNorm1d(num_of_class), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_of_class, num_of_class),
            nn.ReLU(inplace=True),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.cnn_output_size, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096), 
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.network_output_size),
            nn.ReLU(inplace=True)
            ) 

        if self.use_cuda:
            for i in range(len(self.cnn_layers)):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()
            self.cnn_sharing_layer_patch_1_network = self.cnn_sharing_layer_patch_1_network.cuda()
            self.cnn_sharing_layer_patch_3_network = self.cnn_sharing_layer_patch_3_network.cuda()
            self.centroid_identity_layer = self.centroid_identity_layer.cuda()
            self.fully_connected_layer = self.fully_connected_layer.cuda()

        print("MODEL : {}".format(self.name))
    
    def forward(self, input_value):

        if self.use_cuda:
            cnn_input_list = [
                self.cnn_sharing_layer_patch_1_network(input_value["patch_x_scale_1"].float().cuda()),
                self.cnn_sharing_layer_patch_1_network(input_value["patch_y_scale_1"].float().cuda()),
                self.cnn_sharing_layer_patch_1_network(input_value["patch_z_scale_1"].float().cuda()),
                self.cnn_sharing_layer_patch_3_network(input_value["patch_x_scale_3"].float().cuda()),
                self.cnn_sharing_layer_patch_3_network(input_value["patch_y_scale_3"].float().cuda()),
                self.cnn_sharing_layer_patch_3_network(input_value["patch_z_scale_3"].float().cuda()),
                input_value["patch_3d"].float().cuda()
            ]

        else:
            cnn_input_list = [
                self.cnn_sharing_layer_patch_1_network(input_value["patch_x_scale_1"].float()),
                self.cnn_sharing_layer_patch_1_network(input_value["patch_y_scale_1"].float()),
                self.cnn_sharing_layer_patch_1_network(input_value["patch_z_scale_1"].float()),
                self.cnn_sharing_layer_patch_3_network(input_value["patch_x_scale_3"].float()),
                self.cnn_sharing_layer_patch_3_network(input_value["patch_y_scale_3"].float()),
                self.cnn_sharing_layer_patch_3_network(input_value["patch_z_scale_3"].float()),
                input_value["patch_3d"].float()
            ]

        cnn_outputs=[]
        for idx, data in enumerate(cnn_input_list):
            output = self.cnn_layers[idx](data)
            output = output.view(output.size(0), -1)
            cnn_outputs.append(output)

        if self.use_centroid:
            if self.use_cuda:
                centroid = input_value["centroid"].float().cuda()
            else:
                centroid = input_value["centroid"].float()
            if isinstance(self.noise_size, float):
                noise_of_centroid = (torch.randn_like(centroid) * self.noise_size)
                centroid = (centroid + noise_of_centroid).clamp(0,1)

            centroid_output = self.centroid_identity_layer(centroid)
            cnn_outputs.append(centroid_output.view(centroid_output.size(0), -1))

        x = torch.cat(cnn_outputs, 1)

        x = self.fully_connected_layer(x)

        if self.use_cuda:
            x = x.cpu()

        return x

    def get_2d_patch_network(self):
        # outputsize = (29 -4 - 4 - 4) = 13
        return nn.Sequential(
            nn.Conv2d( in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def get_3d_patch_network(self):
        # outpuesize = 13 - 2 - 2 - 2 = 7
        return nn.Sequential(

            nn.Conv3d( in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d( in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
