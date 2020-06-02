import torch
import torch.optim as optim
import sys
import itertools
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from time import localtime, strftime

from apex import amp
from apex.parallel import (DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Engine, Events, _prepare_batch
from ignite.metrics import RunningAverage, Loss
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar

# My modules
from datasets.bdd100k_yolo import BDD100kDataset
from models.yolov3.yolov3 import yolov3_darknet
from utils.tools import get_arguments, get_scheduler
from utils.yolov3.yolo_utils import YOLOLoss, get_optimizer
from utils.yolov3.config import training_params as config
import albumentations as albu
from albumentations.pytorch import ToTensor
from albumentations import HorizontalFlip, Normalize


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
if (args.model == 'yolov3'):
    net = yolov3_darknet(config)

else:
    sys.exit("You did not pick the right script! Exiting...")


is_training = False if config["export_onnx"] else True
net.train(is_training)
# Loading weights if we're using a pretrained model
if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    net.load_state_dict(state_dict, strict=True)
net.to(device)


# Dataset and Train loader
if (args.dataset == 'bdd100k'):
  train_ds = BDD100kDataset(img_size = (config["input_shape"]["width"], config["input_shape"]["height"]))
  # val_ds = BDD100kDataset(transforms = val_tfms,mode = 'val')

if args.distributed:
    kwargs = dict(num_replicas=world_size, rank=local_rank)
    train_sampler = DistributedSampler(train_ds, **kwargs)
    kwargs['shuffle'] = False
    # val_sampler = DistributedSampler(val_dataset, **kwargs)
else:
    train_sampler = None
    # val_sampler = None


train_loader = DataLoader(train_ds,
                          batch_size=args.batch_size,
                          shuffle=True, 
                          num_workers=args.workers, 
                          pin_memory=True)

# Optimizer and scheduler
config.update({"optimizer":{
    "weight_decay": args.weight_decay,
    "type": "sgd",
}})

optimizer = get_optimizer(config, net)
scheduler = get_scheduler(optimizer,args.epochs, args.learning_rate, len(train_loader))

if args.distributed:
    net = convert_syncbn_model(net)
    net = DistributedDataParallel(net)
    
yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLoss(config["priors_info"]["anchors"][i],
                                config["priors_info"]["classes"], 
                                (config["input_shape"]["width"], 
                                 config["input_shape"]["height"])))

 
# Training update function
def update_fn(_trainer, batch):
    """Training function
    Keyword arguments:
    - each bach 
    """
    optimizer.zero_grad()

    images, labels = batch["image"], batch["label"]
    images = images.to(device)
    labels = labels.to(device)

    outputs = net(images)

    
    losses = []
    losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
    for _ in range(len(losses_name)):
        losses.append([])
    for i in range(3):
        _loss_item = yolo_losses[i](outputs[i], labels)
        for j, l in enumerate(_loss_item):
            losses[j].append(l)
    losses = [sum(l) for l in losses]
    loss = losses[0]
        
    loss.backward()
    optimizer.step()

    return {
        'loss': loss 
    }


trainer = Engine(update_fn)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

for name in ['loss']:
    RunningAverage(output_transform=lambda x: x[name]) \
        .attach(trainer, name)
    
# # TODO
# keep 5 best scores in val set
@trainer.on(Events.ITERATION_COMPLETED)
def log_optimizer_params(engine):
    param_groups = optimizer.param_groups[0]
    for h in ['lr', 'momentum', 'weight_decay']:
        if h in param_groups.keys():
            engine.state.metrics[h] = param_groups[h]

if local_rank == 0:
    ProgressBar(persist=True) \
        .attach(trainer, ['loss', 'lr'])
    dirname = strftime("%d-%m-%Y_%Hh%Mm%Ss", localtime())
    dirname = 'checkpoints/' + args.model + '/{}'.format(dirname)
    checkpointer = ModelCheckpoint(
        dirname=dirname,
        filename_prefix=args.model,
        n_saved=10,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpointer,
        to_save={'model': net if not args.distributed else net.module},
    )

trainer.run(train_loader, max_epochs=args.epochs)
