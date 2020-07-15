import argparse
from argparse import ArgumentParser
from ignite.contrib.handlers import (create_lr_scheduler_with_warmup, CosineAnnealingScheduler)
from torch import distributed as dist



def get_arguments():
    """Determines each command lines input and parses them
    """
    parser = ArgumentParser()
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    #Dataset choice
    parser.add_argument(
        "--dataset",
        "-d",
        choices = ['datamatrix', 'coco'],
        default = 'datamatrix',
        help=("The dataset to use to train the model; "
              "Default: datamatrix; "
              "Possible choices datamatrix, coco")
    )

    #Model choice
    parser.add_argument(
        "--model",
        "-m",
        choices = ["faster", "ssd512", "yolov3", "yolov3_spp", "yolov4"],
        default = 'faster',
        help = ("Model to train; "
                "Default: faster; "
                "Possible choices: faster, ssd512, yolov3, yolov3_spp and yolov4")
    )
    
    #backbone choices
    parser.add_argument(
        "--feature_extractor",
        "-feat",
        choices = ["mobilenetv2", "mobilenetv2","resnet50", "resnet50fpn"],
        default = "mobilenetv2",
        help = ("Feature extractor of the model; "
                "Default: mobilenetv2; "
                "Possible choices: mobilenetv2,resnet50, resnet50fpn for faster and mobilenetv2 and resnet50 for ssd")
    )
    parser.add_argument(
        "--pretrained",
        "-pre",
        default=True,
        type=str2bool,
        required=False
        help = ("Torchvision provides COCO weights for Resnet50 FPN and ImageNet weights for Resnet50 and MobileNetV2.")
        )

    # Hyperparameters
    parser.add_argument(
        "--batch_size",
        "-b",
        type = int,
        default = 4,
        help="Batch size value; Default: 4"
    )
    parser.add_argument(
        "--epochs",
        "-eps",
        type = int,
        default = 10,
        help = ("Epochs number; Default: 10")

    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-3,
        help="Learning rate value; Default: 1e-3"
    )
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type = float,
        default = 1e-4,
        help="L2 regularization factor; Default: 1e-4"
    )

    # Definitions
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help = "Number of subprocesses to use for data loading; Default: 4"
    )

    parser.add_argument(
        "--distributed",
        "-dist",
        action='store_true'
    )
    parser.add_argument(
        '--local_rank',
        type=int
    )
    parser.add_argument(
        '--state_dict',
        '-sd', 
        type=str, 
        required=False
    )

    parser.add_argument(
        '--imgs_rect',
        '-rect',
        default=True,
        type=str2bool,
        required=False
    )

    
    
    return parser.parse_args()

def get_scheduler(optimizer,epochs, learning_rate, train_loader_size):
    scheduler =   CosineAnnealingScheduler(optimizer, 
                                          'lr',
                                          learning_rate, 
                                          learning_rate / 100,
                                          epochs * train_loader_size,)
    scheduler = create_lr_scheduler_with_warmup(scheduler, 
                                                0, 
                                                learning_rate, 
                                                100)
    return scheduler




