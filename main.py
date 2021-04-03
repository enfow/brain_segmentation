import argparse
import os

# from model.tested_model import SegNet3DK5L4, SegNet3DK5L3, SegNet3DK5L5, SegNet3DK5L4Cent, SegNet3DK5L4IdentCent, SegNet3DK5L4IdentCentLargeFC
from model.segnet import (SegNet_CNN2A_FC6C, SegNet_CNN3C_FC6C,
                          SegNet_CNN3D_FC7C1, SegNet_CNN4A_FC6C,
                          SegNet_CNN4D_FC5C, SegNet_CNN21D_FC7C1,
                          SegNetOnPaper, SegNetOnPaperNoPool,
                          SegNetOnPaperReplaceCNN)
from train import trainer

present_label_list = [
    0,
    4,
    11,
    15,
    23,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,  # add 33, 34
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    69,
    71,
    72,
    73,
    74,
    75,
    76,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    132,
    133,
    134,
    135,
    136,
    137,  # add 126, 127
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
]  # len(present_label_list) = 143 | all labels on train set

num_of_label = len(present_label_list)


def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument("--seed", default=-1, type=int)
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--lr", default=0.01, type=float)
    p.add_argument("--batch_size", default=512, type=int)
    p.add_argument("--centroid", action='store_true')
    p.add_argument("--centroid_iter", default=1, type=int)
    p.add_argument("--train_size", default=None, type=int)
    p.add_argument("--test_size", default=None, type=int)
    p.add_argument("--noise_size", default=0.01, type=float)
    p.add_argument("--save_img", action="store_true")
    p.add_argument("--use_cuda", action="store_true")

    config = p.parse_args()

    return config


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = define_argparser()
    if config.centroid:

        # SegNet_CNN3C_FC6C
        # SegNet_CNN4D_FC5C
        # SegNet_CNN4D_FC5C

        # SegNet_CNN4D_FC7C1
        # SegNet_CNN3D_FC7C1
        # SegNet_CNN3D_FC7C1
        # SegNet_CNN22D_FC7C1
        # SegNet_CNN21D_FC7C1
        # SegNetOnPaper
        # SegNetOnPaperNoPool
        # SegNetOnPaperReplaceCNN
        model = SegNetOnPaperNoPool(
            num_of_class=num_of_label,
            use_centroid=True,
            use_cuda=config.use_cuda,
            noise_size=config.noise_size,
        )

        centroid_model = SegNetOnPaperNoPool(
            num_of_class=num_of_label,
            use_centroid=False,
            use_cuda=config.use_cuda,
        )

        trainer(
            train_size=config.train_size,
            test_size=config.test_size,
            model=model,
            centroid_model=centroid_model,
            centroid_iter=config.centroid_iter,
            present_label_list=present_label_list,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            seed=config.seed,
            save_img=config.save_img,
            save_model=False,
        )

    else:
        model = SegNet_CNN2A_FC6C(num_of_class=num_of_label, use_cuda=config.use_cuda)

        trainer(
            train_size=config.train_size,
            test_size=config.test_size,
            model=model,
            centroid_model=None,
            present_label_list=present_label_list,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            seed=config.seed,
            save_img=config.save_img,
            save_model=False,
        )

