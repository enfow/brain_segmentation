import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.io import imread

import nibabel as nib


### GET X Y Data ###

def get_data(data_dir, data_file=None):

    """
    data_file -> None : get all data from data directory
    """

    if data_file is None and data_dir:
        file_list = os.listdir(data_dir)
        nii_files = [os.path.join(data_dir, file) for file in file_list if file.endswith(".nii") and not file.endswith("glm.nii")]
        glm_files = []
        for nii in nii_files:
            directory, file = os.path.split(nii)
            glm_files.append(os.path.join(directory, file.split(".")[0]+"_glm.nii"))

    elif isinstance(data_file, str):
        nii_files = [os.path.join(data_dir, data_file)]
        glm_files = [nii_files[0].split(".n")[0] + "_glm.nii"]


    ### settings for nibabel ###
    nib.Nifti1Header.quaternion_threshold = -1e-06

    data_list = [np.pad(np.array(nib.load(data).dataobj).astype(np.float32), ((43,43),(43,43),(43,43)),'constant', constant_values=(0))  for data in nii_files]
    label_list = [np.pad(np.array(nib.load(data).dataobj).astype(np.float32), ((43,43),(43,43),(43,43)),'constant', constant_values=(0)) for data in glm_files]

    # data_list = [np.array(nib.load(data).dataobj).astype(np.float32) for data in nii_files]
    # label_list = [np.array(nib.load(data).dataobj).astype(np.float32) for data in glm_files]

    return data_list, label_list


### CENTROID ###

class RegionCentroids():
    def __init__(self, n_regions):
        self.n_regions = n_regions
        self.barycentres = np.zeros((n_regions, 3))

    def update_barycentres(self, vxs, regions):
        self.barycentres = np.zeros((self.n_regions, 3))
        for i in range(self.n_regions):
            idxs = regions == i
            if vxs[idxs].size == 0:
                continue
            self.barycentres[i] = (np.mean(vxs[idxs], axis=0))

        # For zero values (with no regions present), set them to the mean
        self.barycentres[self.barycentres == 0] = self.barycentres[self.barycentres != 0].mean()

    def compute_scaled_distances(self, vx):
        distances = np.linalg.norm(self.barycentres - vx, axis=1)
        return distances

# region_centroids = RegionCentroids(number_of_label+1) # parameter로 전체 label + 1개 전달
# temp = y.nonzero()
# vxs = np.asarray(temp).T
# region_centroids.update_barycentres(vxs, y[temp])


### GET DATA FOR SINGLE BRAIN
# get centroid object
def get_centroid_obj(y):
    label_list = np.unique(y)
    centroid_obj = RegionCentroids(int(label_list.max())+1) # parameter로 전체 label + 1개 전달
    temp = y.nonzero()
    vxs = np.asarray(temp).T
    centroid_obj.update_barycentres(vxs, y[temp])
    return centroid_obj


# get valid label
def get_valid_label(x, y):
    valid_label = []
    x_shape = x.shape
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            for k in range(x_shape[2]):
                if y[i][j][k] == 0:
                    pass
                else:
                    valid_label.append(
                        {
                            "y_index" : [i,j,k],
                            "y_value" : int(y[i][j][k]),
                        }
                    )
    return valid_label

### TORCH DATALOADER


class BrainSegmentationDataset(Dataset):

    def __init__(self,single_brain_data, valid_label):
        self.single_brain_data = single_brain_data
        self.valid_label = valid_label

    def __len__(self):
        return len(self.valid_label)   # 이거 중요함... 이거 잘못 넣으면 출력되는 데이터 개수가 이상하게 나옴.

    def __getitem__(self, idx):
        x = {}
        data_index = valid_label[idx]["y_index"]
        i, j, k = data_index[0], data_index[1], data_index[2]

        sample = self.single_brain_data[i-43:i+44, j-43:j+44, k-43:k+44]
        x["patch_x_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[43][29:58, 29:58]), 0)
        x["patch_y_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[:, 43][29:58, 29:58]), 0)
        x["patch_z_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[:, : ,43][29:58, 29:58]), 0)

        x["patch_x_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[43]), 0)
        x["patch_y_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[:, 43]), 0)
        x["patch_z_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[:, :, 43]), 0)
        
        # x["centroid"] = self.centroid.compute_scaled_distances(sample["y_index"])
        
        y = valid_label[idx]["y_value"]
        
        return (x, y)


### TEST MODEL AND DATALOADER

if __name__ == "__main__":

    ### GET DATA ###

    print("GET DATA")
    data_dir = "./Dataset/Training/Data"
    data_file = None

    data_list, label_list = get_data(data_dir, data_file=data_file)


    ### GET DATA LOADER ###

    print("GET DATALOADER")
    x, y = data_list[0], label_list[0]

    # number_of_label = 207

    valid_label = get_valid_label(x, y)
    brain_dataset = BrainSegmentationDataset(x, valid_label)

    brain_dataloader = DataLoader(brain_dataset, batch_size=32)


    ### SETTINGS FOR TRAINING ###

    print("SETTINGS FOR TRAINING")
    from model.segnet import SegNet

    learning_rate = 0.01
    epochs = 10

    model = SegNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()


    ### TRAINING ###

    print("TRAINING START")

    step = 0
    loss_list, acc_list = [], []
    for epoch in range(epochs):
        # train
        print('epoch : {}'.format(epoch+1))
        for x, y in brain_dataloader:
            step += 1
            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("step : {}".format(step))

        # validation
        if epoch % 5 == 0:
            for test_data, test_labels in brain_dataloader:
                test_output = model(test_data)
                pred = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred == test_labels.data.numpy()).astype(int).sum()) / float(test_labels.size(0))
                # print("EVALUATION) Epoch : {} | step : {} | loss : {} | accuracy : {}".format(epoch, step, round(float(loss.data),4), accuracy))
                print("EVALUATION) Epoch : {} | step : {} | accuracy : {}".format(epoch, step,  accuracy))
                # loss_list.append(round(float(loss.data), 2))
                acc_list.append(round(float(accuracy), 2))




# def get_one_hot_encoding(label, number_of_label):
#     return np.eye(int(number_of_label) + 1)[int(label)]

# def get_patches(x, y, number_of_label):
#     patches = []
#     x_shape = x.shape
#     one_hot = np.eye(int(number_of_label) + 1)
#     for i in range(x_shape[0]):
#         for j in range(x_shape[1]):
#             for k in range(x_shape[2]):
#                 if y[i][j][k] == 0:
#                     pass
#                 else:
#                     patches.append(
#                         {
#                             "y_index" : [i,j,k],
#                             # "y_value" : one_hot[int(y[i][j][k])],
#                             "y_value" : int(y[i][j][k]),
#                             # "3d_patch_scale_1" : x[i-14:i+15, j-14:j+15, k-14:k+15],
#                             "3d_patch_scale_3" : x[i-43:i+44, j-43:j+44, k-43:k+44],
#                         }
#                     )
#     return patches

# single_brain_data = {
#     "x" : x,
#     "y" : y,
#     "patch" : get_patches(x, y, number_of_label),
#     "centroid" : get_centroid_obj(y),
# }



# with patch data
# class BrainSegmentationDataset(Dataset):

#     def __init__(self,single_brain_data):
#         self.single_brain_data = single_brain_data
#         self.patch = single_brain_data["patch"]
#         self.centroid = single_brain_data["centroid"]
#         print("patch length", len(self.patch))

#     def __len__(self):
#         return len(self.patch)   # 이거 중요함... 이거 잘못 넣으면 출력되는 데이터 개수가 이상하게 나옴.

#     def __getitem__(self, idx):
#         x = {}
#         sample = self.patch[idx]
#         patch_scale_1 = sample["3d_patch_scale_1"]
#         x["patch_x_scale_1"] = torch.unsqueeze(torch.from_numpy(patch_scale_1[14]), 0)
#         x["patch_y_scale_1"] = torch.unsqueeze(torch.from_numpy(patch_scale_1[:, 14]), 0)
#         x["patch_z_scale_1"] = torch.unsqueeze(torch.from_numpy(patch_scale_1[:, : ,14]), 0)
        
#         patch_scale_3 = sample["3d_patch_scale_3"]
#         x["patch_x_scale_3"] = torch.unsqueeze(torch.from_numpy(patch_scale_3[43]), 0)
#         x["patch_y_scale_3"] = torch.unsqueeze(torch.from_numpy(patch_scale_3[:, 43]), 0)
#         x["patch_z_scale_3"] = torch.unsqueeze(torch.from_numpy(patch_scale_3[:, :, 43]), 0)
        
#         x["centroid"] = self.centroid.compute_scaled_distances(sample["y_index"])
        
#         y = sample["y_value"]
        
#         return (x, y)
