import torch
import torch.optim as optim
import sys
import itertools
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from time import localtime, strftime

from apex import amp
from apex.parallel import (DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Events
from ignite.handlers import (global_step_from_engine, ModelCheckpoint)


#object_detection modules
from object_detection.utils.tools import (get_arguments, get_scheduler)
from object_detection.models.yolov3.yolov3_darknet import Darknet









args = get_arguments()

if args.distributed:
    dist.init_process_group('nccl', init_method='env://')
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    local_rank = args.local_rank
else:
    local_rank = 0

torch.cuda.set_device(local_rank)
device = torch.device('cuda')

# Model
if (args.model == "yolov3"):
    model = Darknet(args.yolo_config).to(device)
