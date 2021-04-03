import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from skimage.io import imread
import nibabel as nib

from loader.utils import return_largest_size, add_zero_padding_with_shape, return_label_dicts


def get_data(
            data_dir, 
            data_file=None, 
            num_of_data=None, 
            padding=0, 
            get_test_set=False
            ):

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

    ### MAKE DATA AS SINGLE NDARRAY ###

    data_list = [np.array(nib.load(data).dataobj).astype(np.float32) for data in nii_files]
    label_list = [np.array(nib.load(data).dataobj).astype(np.float32) for data in glm_files]

    if isinstance(num_of_data, int):
        data_list = data_list[:num_of_data + 1]
        label_list = label_list[:num_of_data + 1]

    x_s,y_s,z_s = return_largest_size(data_list)

    data = np.zeros(shape=(len(data_list), x_s+(padding*2), y_s+(padding*2), z_s+(padding*2)))
    label = np.zeros(shape=(len(data_list), x_s+(padding*2), y_s+(padding*2), z_s+(padding*2)))

    for idx, single_data in enumerate(data_list):
        data[idx] = np.pad(add_zero_padding_with_shape(single_data, (x_s,y_s,z_s)), ((padding,padding),(padding,padding),(padding,padding)),'constant', constant_values=(0))
    for idx, single_data in enumerate(label_list):
        label[idx] = np.pad(add_zero_padding_with_shape(single_data, (x_s,y_s,z_s)), ((padding,padding),(padding,padding),(padding,padding)),'constant', constant_values=(0))

    ### PREPROCESSING ###

    # MinMaxScaler
    for num_of_data in range(data.shape[0]):
        scaler = MinMaxScaler()
        for i in range(data[num_of_data].shape[0]):
            scaler.partial_fit(data[num_of_data][i])
        for j in range(data[num_of_data].shape[0]):
            data[num_of_data][j]=scaler.transform(data[num_of_data][j])

    if get_test_set :
        train_set, train_label = data[:-1], label[:-1]
        test_set, test_label = np.array([data[-1]]), np.array([label[-1]])
        del data
        del label
        del data_list
        del label_list

        return train_set, train_label, test_set, test_label

    else : 
        train_set, train_label = data[:-1], label[:-1]
        del data
        del label
        del data_list
        del label_list

        return train_set, train_label



def get_valid_voxel(x, y, label_to_idx):
    valid_voxel = []
    x_shape = x.shape
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            for k in range(x_shape[2]):
                for l in range(x_shape[2]):
                    if y[i][j][k][l] == 0:
                        pass
                    else:
                        valid_voxel.append(
                            {
                                "y_index" : [i,j,k,l],
                                "y_value" : label_to_idx[int(y[i][j][k][l])],
                            }
                        )
    return valid_voxel


class BrainSegmentationDataset(Dataset):

    def __init__(self,single_brain_data, valid_label):
        self.single_brain_data = single_brain_data
        self.valid_label = valid_label
        print("NUM OF PATCHS : {}".format(len(valid_label)))

    def __len__(self):
        return len(self.valid_label)   # 이거 중요함... 이거 잘못 넣으면 출력되는 데이터 개수가 이상하게 나옴.

    def __getitem__(self, idx):
        x = {}
        data_index = self.valid_label[idx]["y_index"]
        i, j, k, l = data_index[0], data_index[1], data_index[2], data_index[3]

        sample = self.single_brain_data[i][j-43:j+44, k-43:k+44, l-43:l+44]
        x["patch_x_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[43][29:58, 29:58]), 0)
        x["patch_y_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[:, 43][29:58, 29:58]), 0)
        x["patch_z_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[:, : ,43][29:58, 29:58]), 0)

        x["patch_x_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[43]), 0)
        x["patch_y_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[:, 43]), 0)
        x["patch_z_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[:, :, 43]), 0)
        
        # x["centroid"] = self.centroid.compute_scaled_distances(sample["y_index"])
        
        y = self.valid_label[idx]["y_value"]
        
        return (x, y)


class BrainSegmentationDataset3D(Dataset):

    def __init__(self,single_brain_data, valid_label):
        self.single_brain_data = single_brain_data
        self.valid_label = valid_label
        print("NUM OF PATCHS : {}".format(len(valid_label)))

    def __len__(self):
        return len(self.valid_label)   # 이거 중요함... 이거 잘못 넣으면 출력되는 데이터 개수가 이상하게 나옴.

    def __getitem__(self, idx):
        x = {}

        data_index = self.valid_label[idx]["y_index"]
        i, j, k, l = data_index[0], data_index[1], data_index[2], data_index[3]

        sample = self.single_brain_data[i][j-43:j+44, k-43:k+44, l-43:l+44]
        x["patch_x_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[43][29:58, 29:58]), 0)
        x["patch_y_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[:, 43][29:58, 29:58]), 0)
        x["patch_z_scale_1"] = torch.unsqueeze(torch.from_numpy(sample[:, : ,43][29:58, 29:58]), 0)

        x["patch_x_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[43]), 0)
        x["patch_y_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[:, 43]), 0)
        x["patch_z_scale_3"] = torch.unsqueeze(torch.from_numpy(sample[:, :, 43]), 0)

        # x["patch_3d"] = torch.unsqueeze(torch.from_numpy(sample[29:58, 29:58, 29:58]), 0)
        x["patch_3d"] = torch.unsqueeze(torch.from_numpy(sample[37:50, 37:50, 37:50]), 0)
        
        y = self.valid_label[idx]["y_value"]
        
        return (x, y)
