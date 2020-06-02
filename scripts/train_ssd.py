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


#My modules
from datasets.bdd100k import BDD100kDataset
from models.ssd.ssd import MobileNetV2SSD_Lite
from utils.prepare_data import get_tfms_ssd300, get_tfms_ssd512
from utils.tools import get_arguments, get_scheduler
from utils.ssd.ssd_utils import MultiboxLoss, MatchPrior, freeze_net_layers
from utils.ssd.transforms_ssd import TrainAugmentation

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

if (args.model == 'ssd300'):
    from utils.ssd import ssd300_config as config
    #train_tfms = get_tfms_ssd300() 

elif  (args.model == 'ssd512'):
    from utils.ssd import ssd512_config as config
    #train_tfms = get_tfms_ssd512()
    
else:
    sys.exit("You did not pick the right script! Exiting...")

model = MobileNetV2SSD_Lite(11, device, model = args.model)
target_transform = MatchPrior(config.priors,
                              config.center_variance,
                              config.size_variance,
                              0.5)

train_tfms = TrainAugmentation(config.image_size, config.image_mean, config.image_std)


if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    params = model.parameters()

elif args.pretrained_model is not None:
    model.init_from_pretrained_ssd(args.pretrained_model)
    freeze_net_layers(model.base_net)
    params = itertools.chain(model.source_layer_add_ons.parameters(), 
                             model.extras.parameters(),
                             model.regression_headers.parameters(), 
                             model.classification_headers.parameters())
    params = [
        {'params': itertools.chain(
            model.source_layer_add_ons.parameters(),
            model.extras.parameters()
        ), 'lr': args.learning_rate},
        {'params': itertools.chain(
            model.regression_headers.parameters(),
            model.classification_headers.parameters()
        )}
    ]
else:
    params = model.parameters() # from scratch
    

model.to(device)

if (args.dataset == 'bdd100k'):
  train_ds = BDD100kDataset(transforms = train_tfms, target_transform = target_transform)
  # val_ds = BDD100kDataset(transforms = val_tfms,mode = 'val')

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
)

optimizer = torch.optim.AdamW(params,
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay,)
scheduler = get_scheduler(optimizer,args.epochs, args.learning_rate, len(train_loader))

criterion = MultiboxLoss(config.priors, 
                         iou_threshold=0.5, 
                         neg_pos_ratio=3, 
                         center_variance=0.1, 
                         size_variance=0.2, 
                         device = device)



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

    images, boxes, labels = batch

    images = images.to(device)
    boxes = boxes.to(device)
    labels = labels.to(device)

    confidence, locations = model(images)

    regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
    loss = regression_loss + classification_loss

    loss.backward()

    optimizer.step()

    return {
        'loss_regression': regression_loss,
        'loss_classifier': classification_loss,
        'loss': loss 
    }


trainer = Engine(update_fn)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

for name in ['loss', 'loss_classifier', 'loss_regression']:
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
        to_save={'model': model if not args.distributed else model.module},
    )

trainer.run(train_loader, max_epochs=args.epochs)


