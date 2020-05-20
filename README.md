# SegNet - 3D brain segmentation with MICCAI 2012 DATASET

- python 3.7.4
- pytorch
- [paper_review](https://enfow.github.io/paper-review/segmentation/2020/04/01/deep_neural_networks_for_anatomical_brain_segmentation/)

## Architecture

**SegNet** is the 3D MRI image segmentation model proposed by Alexandre de Brebisson, Giovanni Montana in 2015. According to the paper, [Deep Neural Networks for Anatomical Brain Segmentation](https://arxiv.org/abs/1502.02445), the SegNet has the following architecture.

![architecture](image/segnet_architecture.png)

It has 8 pathways(six 2D patches, one 3D patch and centroid values of all regions). The model use convolution neural network to extract the features from  2D and 3D patches and use identity layer for centroid vector. After that, concatenate all of the features from each pathways and pass through the fully connected layers to make classification decision. 
 
## Results

On the paper, the dice coefficient score of SegNet with MICCAI 2012 dataset is 0.725 and error rate is 0.163. The reproduction of this git repository code is about error rate of 0.2.

#### Test label image

![test](image/test_label.png)

#### Predict label image

![test](image/pred_label.png)

As you can see, there are label unbalancing issue with the brain region size and it makes that the number of predict label very small. The issue makes not only poor results, but also hindering the learning of the next epoch with trash centroid value. Experimentally these issue tends to occur with larger model(number of layers).
