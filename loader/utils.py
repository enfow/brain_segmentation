import numpy as np


def return_largest_size(data_list):
    x_size, y_size, z_size = 0, 0, 0
    for data in data_list:
        n_x_size, n_y_size, n_z_size = data.shape
        if n_x_size > x_size: x_size = n_x_size
        if n_y_size > y_size: y_size = n_y_size
        if n_z_size > z_size: z_size = n_z_size
    return (x_size, y_size, z_size)


def add_zero_padding_with_shape(arr, shp):
    array_shp = arr.shape
    new_array = np.zeros(shape=shp)
    new_array[:array_shp[0], :array_shp[1], :array_shp[2]] = arr
    return new_array


def return_label_dicts(present_label_list):
    label_to_idx,idx_to_label = {}, {}
    for idx, l in enumerate(present_label_list):
        label_to_idx[l] = idx
        idx_to_label[idx] = l
    return label_to_idx,idx_to_label