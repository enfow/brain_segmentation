import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.dataloader import get_data, get_valid_voxel
from loader.dataloader import BrainSegmentationDataset, BrainSegmentationDataset3D, BrainSegmentationDataset3DCentroid
from loader.utils import return_label_dicts, get_valid_label
from loader.centroid import get_centroid_list


present_label_list =  [  0,   4,  11,  15,  23,  30,  31,  32,  33,  34,  35,  36,  37,     # add 33, 34
                        38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
                        49,  50,  51,  52,  55,  56,  57,  58,  59,  60,  61,
                        62,  63,  64,  65,  66,  69,  71,  72,  73,  74,  75,
                        76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                        123, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137,   # add 126, 127
                        138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
                        149, 150, 151, 152, 153, 154, 155, 156, 157, 160, 161,
                        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
                        173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                        184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196,
                        197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207] # len(present_label_list) = 143 | all labels on train set


### TEST MODEL AND DATALOADER

def define_argparser():

    import argparse

    p = argparse.ArgumentParser()

    p.add_argument(
        '--seed', default=1, type=int
    )
    p.add_argument(
        '--epochs', default=10, type=int
    )
    p.add_argument(
        '--lr', default=0.01, type=float
    )
    p.add_argument(
        '--batch_size', default=512, type=int
    )
    p.add_argument(
        '--centroid', default=True, type=bool
    )

    config = p.parse_args()

    return config


if __name__ == "__main__":

    from model.segnet import SegNet3DK5L4, SegNet3DK5L3, SegNet3DK5L5, SegNet3DK5L4Cent

    import os
    os.environ['CUDA_VISIBLE_DEVICES']="2"

    config = define_argparser()

    # get present label dictionary
    label_to_idx, idx_to_label = return_label_dicts(present_label_list)
    num_of_label = len(label_to_idx.keys())

    seed=config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    with_centroid = config.centroid
    # print(config.centroid)
    # with_centroid = False
    print("With CENTROID : {}".format(with_centroid))
    
    ### GET DATA ###

    print("GET DATA")
    data_dir = "./Dataset/Training/Data"
    data_file = None

    test_dir = "./Dataset/Testing/Data"
    test_file = None

    # get train / test dataset
    # data, label, test_data, test_label = get_data(data_dir, data_file=data_file, num_of_data=None, padding=43, get_test_set=True)

    # get train dataset
    data, label = get_data(data_dir, data_file=data_file, num_of_data=1, padding=43, get_test_set=False)
    label = get_valid_label(label, label_to_idx)
    
    # get test dataset
    test_data, test_label = get_data(test_dir, data_file=test_file, num_of_data=1, padding=43, get_test_set=False)
    test_label = get_valid_label(test_label, label_to_idx)



    print("data shape : {}".format(data.shape) )
    print("test_data shape : {}".format(test_data.shape))

    if with_centroid :  ### WITH CENTROID ###

        model = SegNet3DK5L4Cent(num_of_class=num_of_label, use_cuda=True)

        dir_name = "{}_seed{}".format(model.name, seed)

        if os.path.exists(os.path.join('.', dir_name)):
            raise ValueError("Save Directory is Already Exist")
        else:
            os.mkdir(os.path.join('.', dir_name))

        with open("./{}/log.txt".format(dir_name, model.name, seed), "w") as f:
            f.write("model : {}\n".format(model))
            f.write("data shape : {}\n".format(data.shape))
            f.write("test_data shape : {}\n".format(test_data.shape))

            print("Number of present label : {}".format(num_of_label))
            f.write("Number of present label : {}\n".format(num_of_label))


            print("GET DATALOADER")

            # GET TRAINSET
            valid_voxel = get_valid_voxel(data, label, label_to_idx)
            data_centroid_list = get_centroid_list(label, present_label_list)
            brain_dataset = BrainSegmentationDataset3DCentroid(data, valid_voxel, present_label_list, centroid_list=data_centroid_list)
            brain_dataloader = DataLoader(brain_dataset, batch_size=config.batch_size, shuffle=True)

            test_valid_voxel = get_valid_voxel(test_data, test_label, label_to_idx)

            del data
            del label

            ### SETTINGS FOR TRAINING ###

            print("SETTINGS FOR TRAINING")

            learning_rate = config.lr
            epochs = config.epochs

            print("EPOCH : {} | learning_rate : {}".format(epochs, learning_rate))
            f.write("EPOCH : {} | learning_rate : {} | seed : {}\n".format(epochs, learning_rate, seed))


            # TRAIN TO GET INITIAL CENTROID

            step = 0
            loss_list, acc_list = [], []

            print("INITIALIZE CENTROID FOR TEST SET")
            from model.segnet import SegNet3DK5L4
            centroid_model = SegNet3DK5L4(num_of_class=num_of_label, use_cuda=True) 
            centroid_optimizer = torch.optim.Adam(centroid_model.parameters(), lr=learning_rate)
            centroid_loss_func = nn.CrossEntropyLoss()

            for x, y in tqdm(brain_dataloader, desc="CENTROID TRAIN) "):
                step += 1
                centroid_optimizer.zero_grad()
                output = centroid_model(x)
                loss = centroid_loss_func(output, y)
                loss.backward()
                centroid_optimizer.step()
                loss_list.append(round(float(loss.data), 4))
            del output
            del x
            del y
            f.write("CENTROID TRAIN) step : {} | loss : {}\n".format(step, loss_list[-1] ))

            # test dataloader to get initial centroid
            centroid_brain_dataset = BrainSegmentationDataset3D(test_data, test_valid_voxel)
            centroid_brain_dataloader = DataLoader(centroid_brain_dataset, batch_size=config.batch_size, shuffle=True)

            # test to get centroid
            centroid_model.eval()
            count, accuracy = 0, 0
            pred_test_label = np.zeros_like(test_label)
            for test_x, test_y in tqdm(centroid_brain_dataloader, desc="CENTROID EVALUATION "):
                count += 1
                test_output = centroid_model(test_x)
                pred = torch.max(test_output, 1)[1].data.numpy()
                accuracy += (float((pred == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0)))
                pred_re = pred.reshape(-1, 1)
                pred_idx = test_x["index"].reshape(-1, 4)
                for i in range(pred_re.shape[0]):
                    idx = pred_idx[i]
                    pred_test_label[idx[0],idx[1],idx[2],idx[3]] = pred_re[i]

            print("CENTROID EVALUATION) step : {} | accuracy : {}".format(step,  round( accuracy / count, 4)))
            f.write("CENTROID EVALUATION) step : {} | accuracy : {}\n".format(step,  round( accuracy / count, 4)))

            del test_x
            del test_y
            del centroid_brain_dataset
            del centroid_brain_dataloader
            del centroid_model

            ### TRAINING ###
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_func = nn.CrossEntropyLoss()

            step = 0   
            loss_list, acc_list = [], []
            for epoch in range(epochs):

                # UPDATE TESTSET FOR CENTROID
                test_centroid_list = get_centroid_list(pred_test_label, present_label_list)
                test_brain_dataset = BrainSegmentationDataset3DCentroid(test_data, test_valid_voxel, present_label_list, centroid_list=test_centroid_list)
                test_brain_dataloader = DataLoader(test_brain_dataset, batch_size=config.batch_size, shuffle=True)

                # train
                for x, y in tqdm(brain_dataloader, desc="TRAIN) Epoch {} ".format(epoch+1)):
                    step += 1
                    optimizer.zero_grad()
                    output = model(x)
                    loss = loss_func(output, y)
                    loss.backward()
                    optimizer.step()
                    loss_list.append(round(float(loss.data), 4))
                    if step % 2000 == 0:
                        f.write("TRAIN) Epoch : {} | step : {} | loss : {}\n".format(epoch+1, step, loss_list[-1] ))
                    del output
                    del x
                    del y
                print("TRAIN) Epoch : {} | step : {} | loss : {}".format(epoch+1, step, loss_list[-1] ))
                f.write("TRAIN) Epoch : {} | step : {} | loss : {}\n".format(epoch+1, step, loss_list[-1] ))

                # model save
                # if epoch % 2 == 1:
                torch.save({
                    'epoch' : epoch+1,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict" : optimizer.state_dict(),
                    "loss" : loss,
                }, "./{dir_name}/{model_name}_{epoch}_{seed}.pth".format(dir_name=dir_name, model_name=model.name, epoch=epoch+1, seed=seed))

                # test
                # if epoch % 2 == 1:
                model.eval()
                count, accuracy = 0, 0
                pred_test_label =np.zeros_like(test_label)
                for test_x, test_y in tqdm(test_brain_dataloader, desc="EVALUATION "):
                    pred_idx = test_x["index"].reshape(-1, 4)
                    count += 1
                    test_output = model(test_x)
                    pred = torch.max(test_output, 1)[1].data.numpy()
                    accuracy += (float((pred == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0)))
                    pred_re = pred.reshape(-1, 1)
                    for i in range(pred_re.shape[0]):
                        idx = pred_idx[i]
                        pred_test_label[idx[0],idx[1],idx[2],idx[3]] = pred_re[i]

                print("EVALUATION) Epoch : {} | step : {} | accuracy : {}".format(epoch+1, step,  round( accuracy / count, 4)))
                f.write("EVALUATION) Epoch : {} | step : {} | accuracy : {}\n".format(epoch+1, step,  round( accuracy / count, 4)))
                acc_list.append(round(float(accuracy) / count, 4))

                model.train()

                print("ACCURACY HISTORY : {}".format(acc_list))
                f.write("ACCURACY HISTORY : {}".format(acc_list))


    else :  ### WITHOUT CENTROID ###

        # model = SegNet3DK5L3(num_of_class=num_of_label, use_cuda=True)
        model = SegNet3DK5L4(num_of_class=num_of_label, use_cuda=True) 

        dir_name = "{}_seed{}".format(model.name, seed)

        if os.path.exists(os.path.join('.', dir_name)):
            raise ValueError("Save Directory is Already Exist")
        else:
            os.mkdir(os.path.join('.', dir_name))

        with open("./{}/log.txt".format(dir_name, model.name, seed), "w") as f:
            f.write("model : {}\n".format(model))

            f.write("data shape : {}\n".format(data.shape))
            f.write("test_data shape : {}\n".format(test_data.shape))

            print("Number of present label : {}".format(num_of_label))
            f.write("Number of present label : {}\n".format(num_of_label))
            ### GET DATA LOADER ###

            print("GET DATALOADER")

            # GET TRAINSET
            valid_voxel = get_valid_voxel(data, label, label_to_idx)
            brain_dataset = BrainSegmentationDataset3D(data, valid_voxel)
            brain_dataloader = DataLoader(brain_dataset, batch_size=config.batch_size, shuffle=True)

            # GET TESTSET
            test_valid_voxel = get_valid_voxel(test_data, test_label, label_to_idx)
            test_brain_dataset = BrainSegmentationDataset3D(test_data, test_valid_voxel)
            test_brain_dataloader = DataLoader(test_brain_dataset, batch_size=config.batch_size, shuffle=True)

            del data
            del label


            ### SETTINGS FOR TRAINING ###

            print("SETTINGS FOR TRAINING")

            learning_rate = config.lr
            epochs = config.epochs

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_func = nn.CrossEntropyLoss()

            print("EPOCH : {} | learning_rate : {}".format(epochs, learning_rate))
            f.write("EPOCH : {} | learning_rate : {} | seed : {}\n".format(epochs, learning_rate, seed))

            print("TRAINING START")

            step = 0
            loss_list, acc_list = [], []
            for epoch in range(epochs):
                # train
                for x, y in tqdm(brain_dataloader, desc="TRAIN) Epoch {} ".format(epoch+1)):
                    step += 1
                    optimizer.zero_grad()
                    output = model(x)
                    loss = loss_func(output, y)
                    loss.backward()
                    optimizer.step()
                    loss_list.append(round(float(loss.data), 4))
                    if step % 2000 == 0:
                        # print("TRAIN) Epoch : {} | step : {} | loss : {}".format(epoch+1, step, loss_list[-1] ))
                        f.write("TRAIN) Epoch : {} | step : {} | loss : {}\n".format(epoch+1, step, loss_list[-1] ))
                    del output
                    del x
                    del y
                print("TRAIN) Epoch : {} | step : {} | loss : {}".format(epoch+1, step, loss_list[-1] ))
                f.write("TRAIN) Epoch : {} | step : {} | loss : {}\n".format(epoch+1, step, loss_list[-1] ))

                # model save
                # if epoch % 2 == 1:
                torch.save({
                    'epoch' : epoch+1,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict" : optimizer.state_dict(),
                    "loss" : loss,
                }, "./{dir_name}/{model_name}_{epoch}_{seed}.pth".format(dir_name=dir_name, model_name=model.name, epoch=epoch+1, seed=seed))

                # test
                # if epoch % 2 == 1:
                model.eval()
                count, accuracy = 0, 0
                for test_data, test_labels in tqdm(test_brain_dataloader, desc="EVALUATION "):
                    count += 1
                    test_output = model(test_data)
                    pred = torch.max(test_output, 1)[1].data.numpy()
                    accuracy += (float((pred == test_labels.data.numpy()).astype(int).sum()) / float(test_labels.size(0)))
                    del test_data
                    del test_labels
                    del test_output       

                print("EVALUATION) Epoch : {} | step : {} | accuracy : {}".format(epoch+1, step,  round( accuracy / count, 4)))
                f.write("EVALUATION) Epoch : {} | step : {} | accuracy : {}\n".format(epoch+1, step,  round( accuracy / count, 4)))
                acc_list.append(round(float(accuracy) / count, 4))
                model.train()

                print("ACCURACY HISTORY : {}".format(acc_list))
                f.write("ACCURACY HISTORY : {}".format(acc_list))


# seed 1 -> no centroid data 10 | test 2
# seed 2 -> centroid data 10 | test 2
# seed 3 -> no centroid data ALL | test 5
# seed 6 -> no centroid data All | test 5