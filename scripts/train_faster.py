import torch
import torch.optim as optim
from torch import distributed as dist
from torch.utils.data import (DataLoader, DistributedSampler)
from time import localtime, strftime
from itertools import chain
import sys

from apex.parallel import (DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Events
from ignite.handlers import (global_step_from_engine, ModelCheckpoint)

# object_detection modules
from object_detection.datasets.datamatrix import DataMatrixDataset
from object_detection.datasets.coco import COCODetection

from object_detection.models.faster.faster_rcnn import (resnet50fpn_fasterRCNN, 
                                                        resnet50_fasterRCNN, 
                                                        mobilenetv2_fasterRCNN)
from object_detection.utils.prepare_data import (get_tfms_faster,
                                                 collate_fn)
from object_detection.utils.tools import (get_arguments, get_scheduler)
from object_detection.engine import (create_detection_trainer, create_detection_evaluator)
from object_detection.utils.evaluation import convert_to_coco_api


args = get_arguments()

if args.distributed:
    dist.init_process_group("nccl", init_method="env://")
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    local_rank = args.local_rank
else:
    local_rank = 0

torch.cuda.set_device(local_rank)
device = torch.device("cuda")

if args.dataset == "datamatrix":
    train_tfms, val_tfms = get_tfms_faster(ds = "datamatrix")
    train_ds = DataMatrixDataset(transforms = train_tfms)
    val_ds = DataMatrixDataset(transforms = val_tfms, mode = 'val')
    n_classes = 2
elif args.dataset == "coco":
    train_tfms, val_tfms = get_tfms_faster(ds = "coco")
    train_ds = COCODetection(transforms = train_tfms)
    val_ds = COCODetection(transforms = val_tfms, mode = 'val')
    n_classes = 91


if args.distributed:
    kwargs = dict(num_replicas=world_size, rank=local_rank)
    train_sampler = DistributedSampler(train_ds, **kwargs)
    kwargs["shuffle"] = False
    val_sampler = DistributedSampler(val_ds, **kwargs)
else:
    train_sampler = None
    val_sampler = None

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=not args.distributed,
    drop_last=True,
    num_workers=args.workers,
    sampler=train_sampler,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.workers,
    sampler=val_sampler,
    collate_fn=collate_fn
)
coco_api_val_dataset = convert_to_coco_api(val_ds)

if (args.model == "faster"):
    if (args.feature_extractor == "mobilenetv2"):
        model = mobilenetv2_fasterRCNN(n_classes)
    elif (args.feature_extractor == "resnet50fpn"):
        model = resnet50fpn_fasterRCNN(n_classes)
    elif (args.feature_extractor == "resnet50"):
        model = resnet50_fasterRCNN(n_classes)
else:
    sys.exit("You did not pick the right script! Exiting...")



if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location = "cpu")
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


evaluator = create_detection_evaluator(args.model,
                                       model, 
                                       device, 
                                       coco_api_val_dataset
                                       )

trainer = create_detection_trainer(args.model, 
                                   model, 
                                   optimizer, 
                                   device,
                                   val_loader,
                                   evaluator,
                                   loss_fn = None,
                                   logging = local_rank == 0
                                   )

trainer.add_event_handler(
    Events.ITERATION_COMPLETED, scheduler,
)


if local_rank == 0:
    dirname = strftime("%d-%m-%Y_%Hh%Mm%Ss", localtime())
    dirname = "checkpoints/" + args.dataset + "/" + args.feature_extractor + args.model + "/{}".format(dirname)
    
    checkpointer = ModelCheckpoint(
        dirname=dirname,
        filename_prefix=args.model,
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpointer,
        to_save={'model': model if not args.distributed else model.module},
    )

trainer.run(train_loader, max_epochs=args.epochs)
