import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.centroid import get_centroid_list, get_updated_centroid_list
from loader.dataloader import (BrainSegmentationDataset,
                               BrainSegmentationDataset3D,
                               BrainSegmentationDataset3DCentroid, get_data,
                               get_valid_voxel)
from loader.utils import get_valid_label, return_label_dicts


def trainer(
    train_size=None,
    test_size=5,
    model=None,
    centroid_model=None,
    centroid_iter=1,
    present_label_list=None,
    epochs=5,
    batch_size=512,
    lr=0.01,
    seed=-1,
    save_img=True,
    save_model=False,
):

    if seed == -1:
        seed = np.random.randint(9999)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if model is None:
        raise ValueError("No model")

    if centroid_model is None:
        use_centroid = False
    else:
        use_centroid = True
    print("use centroid ", use_centroid)

    if present_label_list is None:
        raise ValueError("No present_label_list")

    if train_size is None:
        dsize = "f"
    else:
        dsize = train_size
    dir_name = "{}_dsize_{}_citer{}_batch{}_seed{}".format(
        model.name, dsize, centroid_iter, batch_size, seed
    )

    if os.path.exists(os.path.join(".", dir_name)):
        raise ValueError("Save Directory is Already Exist")
    else:
        os.mkdir(os.path.join(".", dir_name))

    with open("./{}/log.txt".format(dir_name, model.name, seed), "w") as f:

        f.write("model : {}\n".format(model))

        label_to_idx, idx_to_label = return_label_dicts(present_label_list)

        data_dir = "./Dataset/Training/Data"
        data_file = None

        test_dir = "./Dataset/Testing/Data"
        test_file = None

        # get train dataset
        print()
        print("GET TRAIN DATA")
        data, label, min_value, max_value = get_data(
            data_dir,
            data_file=data_file,
            num_of_data=train_size,
            padding=43,
            get_test_set=False,
        )
        label = get_valid_label(label, label_to_idx)
        print("train_data shape: {}".format(data.shape))
        print("Number of train_label: {}".format(len(np.unique(label.reshape(-1)))))
        f.write("train_data shape: {}\n".format(data.shape))
        f.write("Number of train_label: {}\n".format(len(np.unique(label.reshape(-1)))))

        # get test dataset
        print()
        print("GET TEST DATA")
        test_data, test_label, _, _ = get_data(
            test_dir,
            data_file=test_file,
            num_of_data=test_size,
            padding=43,
            get_test_set=False,
            min_value=min_value,
            max_value=max_value,
        )
        test_label = get_valid_label(test_label, label_to_idx)

        print("test_data shape: {}".format(test_data.shape))
        print("Number of test_label: ", len(np.unique(test_label.reshape(-1))))
        f.write("test_data shape: {}\n".format(test_data.shape))
        f.write(
            "Number of test_label: {}\n".format(len(np.unique(test_label.reshape(-1))))
        )

        if save_img:
            save_label(test_label, dir_name, "test_label")

        if use_centroid:

            print()
            print("GET DATALOADER")

            # GET TRAINSET
            valid_voxel = get_valid_voxel(data, label, label_to_idx)
            data_centroid_list = get_centroid_list(label, present_label_list)
            brain_dataset = BrainSegmentationDataset3DCentroid(
                data,
                valid_voxel,
                present_label_list,
                centroid_list=data_centroid_list,
                is_test=True,
            )
            brain_dataloader = DataLoader(
                brain_dataset, batch_size=batch_size, shuffle=True
            )

            test_valid_voxel = get_valid_voxel(test_data, test_label, label_to_idx)

            del data
            del label

            ### SETTINGS FOR TRAINING ###
            print()
            print("SETTINGS FOR TRAINING")

            learning_rate = lr

            print(
                "EPOCH : {} | learning_rate : {} | seed : {}\n".format(
                    epochs, learning_rate, seed
                )
            )
            f.write(
                "EPOCH : {} | learning_rate : {} | seed : {}\n".format(
                    epochs, learning_rate, seed
                )
            )

            # TRAIN TO GET INITIAL CENTROID

            step = 0
            loss_list, acc_list = [], []

            print("INITIALIZE CENTROID FOR TEST SET")
            centroid_optimizer = torch.optim.Adam(
                centroid_model.parameters(), lr=learning_rate
            )
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
            print("CENTROID TRAIN) step : {} | loss : {}\n".format(step, loss_list[-1]))
            f.write(
                "CENTROID TRAIN) step : {} | loss : {}\n".format(step, loss_list[-1])
            )

            # test dataloader to get initial centroid
            centroid_brain_dataset = BrainSegmentationDataset3D(
                test_data, test_valid_voxel
            )
            centroid_brain_dataloader = DataLoader(
                centroid_brain_dataset, batch_size=batch_size, shuffle=True
            )

            # test to get centroid
            centroid_model.eval()
            count, accuracy = 0, 0
            pred_test_label = np.zeros_like(test_label)
            for test_x, test_y in tqdm(
                centroid_brain_dataloader, desc="CENTROID EVALUATION "
            ):
                count += 1
                test_output = centroid_model(test_x)
                pred = torch.max(test_output, 1)[1].data.numpy()
                accuracy += float(
                    (pred == test_y.data.numpy()).astype(int).sum()
                ) / float(test_y.size(0))
                pred_re = pred.reshape(-1, 1)
                pred_idx = test_x["index"].reshape(-1, 4)
                for i in range(pred_re.shape[0]):
                    idx = pred_idx[i]
                    pred_test_label[idx[0], idx[1], idx[2], idx[3]] = pred_re[i]

            if save_img:
                save_label(pred_test_label, dir_name, "pred_test_label_init")

            print(
                "CENTROID EVALUATION) step : {} | accuracy : {}".format(
                    step, round(accuracy / count, 4)
                )
            )
            f.write(
                "CENTROID EVALUATION) step : {} | accuracy : {}\n".format(
                    step, round(accuracy / count, 4)
                )
            )

            del test_x
            del test_y
            del centroid_brain_dataset
            del centroid_brain_dataloader
            del centroid_model

            ### TRAINING ###
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_func = nn.CrossEntropyLoss()

            test_centroid_list = get_centroid_list(pred_test_label, present_label_list)

            print()
            step = 0
            centroid_step = 0
            loss_list, acc_list = [], []
            average_loss_size = 1000
            for epoch in range(epochs):

                # UPDATE TESTSET FOR CENTROID
                print(
                    "Number of pred_test_label",
                    len(np.unique(pred_test_label.reshape(-1))),
                )
                f.write(
                    "Number of pred_test_label : {}\n".format(
                        len(np.unique(pred_test_label.reshape(-1)))
                    )
                )
                # if epoch > 0:
                #     test_centroid_list = get_updated_centroid_list(pred_test_label, test_centroid_list)
                # test_brain_dataset = BrainSegmentationDataset3DCentroid(test_data, test_valid_voxel, present_label_list, centroid_list=test_centroid_list, is_test=True)
                # test_brain_dataloader = DataLoader(test_brain_dataset, batch_size=batch_size)

                # train
                for x, y in tqdm(
                    brain_dataloader, desc="TRAIN) Epoch {} ".format(epoch + 1)
                ):
                    step += 1
                    optimizer.zero_grad()
                    output = model(x)
                    loss = loss_func(output, y)
                    loss.backward()
                    optimizer.step()
                    loss_list.append(round(float(loss.data), 4))
                    if step % average_loss_size == 0:
                        average_loss = round(
                            np.sum(loss_list[-average_loss_size:]) / average_loss_size,
                            4,
                        )
                        f.write(
                            "TRAIN) Epoch : {} | step : {} | ave_loss : {}\n".format(
                                epoch + 1, step, average_loss
                            )
                        )
                    del output
                    del x
                    del y
                print(
                    "TRAIN) Epoch : {} | step : {} | loss : {}".format(
                        epoch + 1, step, loss_list[-1]
                    )
                )
                f.write(
                    "TRAIN) Epoch : {} | step : {} | loss : {}\n".format(
                        epoch + 1, step, loss_list[-1]
                    )
                )

                # model save
                # if epoch % 2 == 1:
                if save_model:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        "./{dir_name}/{model_name}_{epoch}_{seed}.pth".format(
                            dir_name=dir_name,
                            model_name=model.name,
                            epoch=epoch + 1,
                            seed=seed,
                        ),
                    )

                # test
                # if epoch % 2 == 1:
                model.eval()
                count, accuracy = 0, 0
                for cent_iter in range(centroid_iter):
                    centroid_step += 1
                    if (epoch > 0) or (cent_iter > 0):
                        test_centroid_list = get_updated_centroid_list(
                            pred_test_label, test_centroid_list
                        )
                    test_brain_dataset = BrainSegmentationDataset3DCentroid(
                        test_data,
                        test_valid_voxel,
                        present_label_list,
                        centroid_list=test_centroid_list,
                        is_test=True,
                    )
                    test_brain_dataloader = DataLoader(
                        test_brain_dataset, batch_size=batch_size
                    )

                    pred_test_label = np.zeros_like(test_label)
                    for test_x, test_y in tqdm(
                        test_brain_dataloader, desc="EVALUATION "
                    ):
                        pred_idx = test_x["index"].reshape(-1, 4)
                        count += 1
                        test_output = model(test_x)
                        pred = torch.max(test_output, 1)[1].data.numpy()
                        accuracy += float(
                            (pred == test_y.data.numpy()).astype(int).sum()
                        ) / float(test_y.size(0))
                        pred_re = pred.reshape(-1, 1)
                        for i in range(pred_re.shape[0]):
                            idx = pred_idx[i]
                            pred_test_label[idx[0], idx[1], idx[2], idx[3]] = pred_re[i]

                    if save_img:
                        save_number = "{}_{}".format(epoch + 1, centroid_step)
                        save_label(
                            pred_test_label,
                            dir_name,
                            "pred_test_label",
                            save_label=save_number,
                        )

                    print(
                        "EVALUATION) Epoch : {} | step : {} | centroid_iter : {} | accuracy : {}".format(
                            epoch + 1, step, centroid_step, round(accuracy / count, 4)
                        )
                    )
                    f.write(
                        "EVALUATION) Epoch : {} | step : {} | centroid_iter : {} | accuracy : {}\n".format(
                            epoch + 1, step, centroid_step, round(accuracy / count, 4)
                        )
                    )
                    acc_list.append(round(float(accuracy) / count, 4))

                model.train()

                print("ACCURACY HISTORY : {}\n".format(acc_list))
                f.write("ACCURACY HISTORY : {}\n".format(acc_list))

        else:

            # GET TRAINSET
            valid_voxel = get_valid_voxel(data, label, label_to_idx)
            brain_dataset = BrainSegmentationDataset3D(data, valid_voxel)
            brain_dataloader = DataLoader(
                brain_dataset, batch_size=batch_size, shuffle=True
            )

            # GET TESTSET
            test_valid_voxel = get_valid_voxel(test_data, test_label, label_to_idx)
            test_brain_dataset = BrainSegmentationDataset3D(test_data, test_valid_voxel)
            test_brain_dataloader = DataLoader(
                test_brain_dataset, batch_size=batch_size
            )

            del data
            del label

            ### SETTINGS FOR TRAINING ###

            print("SETTINGS FOR TRAINING")

            learning_rate = lr
            epochs = epochs

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_func = nn.CrossEntropyLoss()

            print(
                "EPOCH : {} | learning_rate : {} | seed : {}\n".format(
                    epochs, learning_rate, seed
                )
            )
            f.write(
                "EPOCH : {} | learning_rate : {} | seed : {}\n".format(
                    epochs, learning_rate, seed
                )
            )

            print("TRAINING START")

            step = 0
            loss_list, acc_list = [], []
            average_loss_size = 1000
            for epoch in range(epochs):
                # train
                for x, y in tqdm(
                    brain_dataloader, desc="TRAIN) Epoch {} ".format(epoch + 1)
                ):
                    step += 1
                    optimizer.zero_grad()
                    output = model(x)
                    loss = loss_func(output, y)
                    loss.backward()
                    optimizer.step()
                    loss_list.append(round(float(loss.data), 4))
                    if step % average_loss_size == 0:
                        average_loss = round(
                            np.sum(loss_list[-average_loss_size:]) / average_loss_size,
                            4,
                        )
                        f.write(
                            "TRAIN) Epoch : {} | step : {} | loss : {}\n".format(
                                epoch + 1, step, average_loss
                            )
                        )
                    del output
                    del x
                    del y
                print(
                    "TRAIN) Epoch : {} | step : {} | loss : {}".format(
                        epoch + 1, step, loss_list[-1]
                    )
                )
                f.write(
                    "TRAIN) Epoch : {} | step : {} | loss : {}\n".format(
                        epoch + 1, step, loss_list[-1]
                    )
                )

                # model save
                # if epoch % 2 == 1:
                if save_model:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        "./{dir_name}/{model_name}_{epoch}_{seed}.pth".format(
                            dir_name=dir_name,
                            model_name=model.name,
                            epoch=epoch + 1,
                            seed=seed,
                        ),
                    )

                # test
                # if epoch % 2 == 1:
                model.eval()
                count, accuracy = 0, 0
                pred_test_label = np.zeros_like(test_label)
                for test_data, test_labels in tqdm(
                    test_brain_dataloader, desc="EVALUATION "
                ):
                    pred_idx = test_data["index"].reshape(-1, 4)
                    count += 1
                    test_output = model(test_data)
                    pred = torch.max(test_output, 1)[1].data.numpy()
                    accuracy += float(
                        (pred == test_labels.data.numpy()).astype(int).sum()
                    ) / float(test_labels.size(0))
                    pred_re = pred.reshape(-1, 1)
                    for i in range(pred_re.shape[0]):
                        idx = pred_idx[i]
                        pred_test_label[idx[0], idx[1], idx[2], idx[3]] = pred_re[i]
                    del test_data
                    del test_labels
                    del test_output

                if save_img:
                    save_label(
                        pred_test_label,
                        dir_name,
                        "pred_test_label",
                        save_label=epoch + 1,
                    )

                print(
                    "Number of pred_test_label",
                    len(np.unique(pred_test_label.reshape(-1))),
                )
                f.write(
                    "Number of pred_test_label : {}\n".format(
                        len(np.unique(pred_test_label.reshape(-1)))
                    )
                )

                print(
                    "EVALUATION) Epoch : {} | step : {} | accuracy : {}".format(
                        epoch + 1, step, round(accuracy / count, 4)
                    )
                )
                f.write(
                    "EVALUATION) Epoch : {} | step : {} | accuracy : {}\n".format(
                        epoch + 1, step, round(accuracy / count, 4)
                    )
                )
                acc_list.append(round(float(accuracy) / count, 4))
                model.train()

                print("ACCURACY HISTORY : {}\n".format(acc_list))
                f.write("ACCURACY HISTORY : {}\n".format(acc_list))


def save_label(label_data, dir_name, save_name, save_label=None):
    if save_label is not None:
        save_name = "{}_{}".format(save_name, save_label)

    fig, axarr = plt.subplots(1, 4, figsize=(35, 25))
    axarr[0].imshow(label_data[0][125][50:250, 100:300])
    axarr[1].imshow(label_data[0][150][50:250, 100:300])
    axarr[2].imshow(label_data[0][175][50:250, 100:300])
    axarr[3].imshow(label_data[0][200][50:250, 100:300])

    plt.savefig("./{}/{}.png".format(dir_name, save_name), dpi=200)
    plt.close(fig)

    with open("./{}/{}.pkl".format(dir_name, save_name), "wb") as f:
        pickle.dump(label_data, f)
