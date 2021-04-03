import numpy as np


class RegionCentroids:
    def __init__(self, n_regions):
        self.n_regions = n_regions
        self.barycentres = np.zeros((n_regions, 3))

    def update_barycentres(self, vxs, regions):
        self.barycentres = np.zeros((self.n_regions, 3))
        for i in range(self.n_regions):
            idxs = regions == i
            if vxs[idxs].size == 0:
                continue
            self.barycentres[i] = np.mean(vxs[idxs], axis=0)

        # For zero values (with no regions present), set them to the mean
        self.barycentres[self.barycentres == 0] = self.barycentres[
            self.barycentres != 0
        ].mean()

    def compute_scaled_distances(self, vx):
        distances = np.linalg.norm(self.barycentres - vx, axis=1)
        max_distance = distances.max()
        distances = distances / max_distance
        return distances


def get_centroid_obj(single_brain_label, present_label_list):
    # centroid_obj = RegionCentroids(int(present_label_list.max())+1) # parameter로 전체 label + 1개 전달
    centroid_obj = RegionCentroids(
        len(present_label_list)
    )  # parameter로 전체 label + 1개 전달
    temp = single_brain_label.nonzero()
    vxs = np.asarray(temp).T
    centroid_obj.update_barycentres(vxs, single_brain_label[temp])
    return centroid_obj


def get_centroid_list(label, present_label_list):
    centroid_list = []
    num_of_brain = label.shape[0]
    for idx in range(num_of_brain):
        centroid_list.append(get_centroid_obj(label[idx], present_label_list))
    return centroid_list


def get_updated_centroid_list(updated_label, centroid_list):
    updated_centroid_list = centroid_list
    num_of_brain = updated_label.shape[0]
    for idx in range(num_of_brain):
        temp = updated_label[idx].nonzero()
        vxs = np.asarray(temp).T
        updated_centroid_list[idx].update_barycentres(vxs, updated_label[idx][temp])
    return updated_centroid_list
