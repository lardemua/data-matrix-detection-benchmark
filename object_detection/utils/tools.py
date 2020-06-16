import argparse
from argparse import ArgumentParser
from ignite.contrib.handlers import (create_lr_scheduler_with_warmup, CosineAnnealingScheduler)
from torch import distributed as dist



def get_arguments():
    """Determines each command lines input and parses them
    """
    parser = ArgumentParser()

    #Dataset choice
    parser.add_argument(
        "--dataset",
        "-d",
        choices = ['bdd100k', 'datamatrix'],
        default = 'datamatrix',
        help=("The dataset to use to train the model; "
              "Default: bdd100k; "
              "Possible choices: bdd100k")
    )

    #Model choice
    parser.add_argument(
        "--model",
        "-m",
        choices = ["faster", "ssd512", "yolov3"],
        default = 'faster',
        help = ("Model to train; "
                "Default: faster; "
                "Possible choices: faster-rcnn, ssd512 and yolov3")
    )
    
    #backbone choice
    parser.add_argument(
        "--feature_extractor",
        "-feat",
        choices = ["mobilenetv2", "mobilenetv2fpn","resnet50", "resnet50fpn"],
        default = "mobilenetv2",
        help = ("Feature extractor of the model; "
                "Default: mobilenetv2; "
                "Possible choices: mobilenetv2, mobilenetv2fpn,resnet50, resnet50fpn for now")
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
        "--device",
        "-dev",
        default = 'cuda:0',
        help = "Device on which the network will be trained; Default: cuda:2"
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
        '--pretrained_model',
        '-pm',
        type=str,
        required=False
    )
    
    parser.add_argument(
        '--yolo_config',
        '-cfg',
        default='object_detection/utils/yolo/yolov3_spp.cfg',
        type=str,
        required=False
    )
    parser.add_argument(
    '--imgs_rect',
    '-rect',
    default=True,
    type=bool,
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




