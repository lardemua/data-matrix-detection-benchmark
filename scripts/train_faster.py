import torch
import torch.optim as optim
import sys
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from time import localtime, strftime

from apex import amp
from apex.parallel import (DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Engine, Events, _prepare_batch
from ignite.metrics import RunningAverage, Loss
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar


#My modules
from datasets.datamatrix import DataMatrixDataset
from models.faster_rcnn import resnet50fpn_fasterRCNN, resnet50_fasterRCNN, mobilenetv2_fasterRCNN
from utils.prepare_data import get_tfms_faster,collate_fn, transform_inputs
from utils.tools import get_arguments, get_scheduler


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


train_tfms, val_tfms = get_tfms_faster()
if args.dataset == 'datamatrix':
  train_ds = DataMatrixDataset(transforms = train_tfms)

if args.distributed:
    kwargs = dict(num_replicas=world_size, rank=local_rank)
    train_sampler = DistributedSampler(train_ds, **kwargs)
    kwargs['shuffle'] = False
    # val_sampler = DistributedSampler(val_dataset, **kwargs)
else:
    train_sampler = None
    # val_sampler = None

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=not args.distributed,
    drop_last=True,
    num_workers=args.workers,
    sampler=train_sampler,
    collate_fn=collate_fn,
)

# val_loader = DataLoader(
#     val_ds,
#     batch_size=4,
#     shuffle=False,
#     drop_last=False,
#     num_workers=8,
#     sampler=val_sampler,
# )


if (args.model == 'faster'):
    if (args.feature_extractor == 'mobilenetv2'):
        model = mobilenetv2_fasterRCNN(2)
    elif (args.feature_extractor == 'resnet50fpn'):
        model = resnet50fpn_fasterRCNN(2)
    elif (args.feature_extractor == 'resnet50'):
        model = resnet50_fasterRCNN(2)
else:
    sys.exit("You did not pick the right script! Exiting...")



if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)
scheduler = get_scheduler(optimizer,args.epochs, args.learning_rate, len(train_loader))

if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)


def update_fn(_trainer, batch):
    """Training function
    Keyword arguments:
    - each bach 
    """
    model.train()
    optimizer.zero_grad()

    images, targets = batch
    images, targets = transform_inputs(images, targets, device)
    
    losses = model(images, targets)
    loss = sum([loss for loss in losses.values()])

    loss.backward()

    optimizer.step()
    return {
        'loss': loss.item(),
        'loss_classifier':  losses['loss_classifier'].item(), 
        'loss_box_reg':     losses['loss_box_reg'].item(),
        'loss_objectness':  losses['loss_objectness'].item(),
        'loss_rpn_box_reg': losses['loss_rpn_box_reg'].item(),
    }


trainer = Engine(update_fn)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)


for name in ['loss', 'loss_classifier']:
    RunningAverage(output_transform=lambda x: x[name]) \
        .attach(trainer, name)

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
    dirname = 'checkpoints/' + args.feature_extractor + args.model + '/{}'.format(dirname)
    checkpointer = ModelCheckpoint(
        dirname=dirname,
        filename_prefix=args.model,
        n_saved=10,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpointer,
        to_save={'model': model if not args.distributed else model.module},
    )

trainer.run(train_loader, max_epochs=args.epochs)
