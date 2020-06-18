import torch
import torch.optim as optim
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim.lr_scheduler as lr_scheduler
import sys
import itertools
from time import localtime, strftime

from apex import amp
from apex.parallel import (DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Events
from ignite.handlers import (global_step_from_engine, ModelCheckpoint)


#object_detection modules
from object_detection.utils.tools import (get_arguments, get_scheduler)
from object_detection.models.yolov3.yolov3_darknet import Darknet
from object_detection.datasets.datamatrix_yolo import DataMatrixDataset
from object_detection.engine import (create_detection_trainer, create_detection_evaluator)
from object_detection.utils.evaluation import convert_to_coco_api 
from object_detection.losses.yolo_loss import compute_loss as loss_fn
from object_detection.utils.prepare_data import collate_fn 

# Hyperparameters (results68: 59.9 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54, #1.0,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 205.76,  #64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20, #0.225,  # iou training threshold
       'lr0': 0.01, #0.001 # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,#-4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)


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
if (args.model == "yolov3" or args.model == "yolov3_spp" or args.model == "yolov4"):
    if args.model == "yolov3":
        yolo_config = "object_detection/utils/yolo/yolov3.cfg"
    elif args.model == "yolov3_spp":
        yolo_config = "object_detection/utils/yolo/yolov3_spp.cfg"
    else: #yolov4
        yolo_config = "object_detection/utils/yolo/yolov4.cfg"
        
    model = Darknet(yolo_config)

if args.state_dict is not None:
    chkpt = torch.load(args.state_dict, map_location="cpu")
    model.load_state_dict(chkpt, strict=False)
    
model.to(device)

if (args.dataset == 'datamatrix'):
    # training set 
    train_ds = DataMatrixDataset(mode = "train", 
                                img_size = 1024, 
                                batch_size = args.batch_size,
                                augment = True,
                                hyp = hyp,  # augmentation hyperparameters
                                rect = args.imgs_rect)


    # validation set
    val_ds = DataMatrixDataset(mode = "val",
                            img_size = 1024,
                            batch_size = args.batch_size,
                            hyp = hyp,
                            rect = True)
    model.nc = 1
    model.hyp = hyp 
    model.gr = 0.0

# training set dataloader
train_loader = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        shuffle=not args.imgs_rect,  # Shuffle=True unless rectangular training is used
                        pin_memory=True,
                        collate_fn=train_ds.collate_fn)

# validatation set dataloader
val_loader = DataLoader(val_ds,
                        batch_size = args.batch_size,
                        num_workers = args.workers,
                        pin_memory = True,
                        collate_fn = collate_fn)

if args.distributed:
    kwargs = dict(num_replicas=world_size, rank=local_rank)
    train_sampler = DistributedSampler(train_ds, **kwargs)
    kwargs['shuffle'] = False
    val_sampler = DistributedSampler(val_ds, **kwargs)
else:
    train_sampler = None
    val_sampler = None

coco_api_val_dataset = convert_to_coco_api(val_ds)

#Optimizer    
pg0, pg1, pg2 = [], [], [] # optimizer parameter groups
for k, v in dict(model.named_parameters()).items():
    if ".bias" in k:
        pg2 += [v]
    elif "Conv2d.weight" in k:
        pg1 += [v]
    else:
        pg0 += [v]    
optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
del pg0, pg1, pg2

#Scheduler 
scheduler = get_scheduler(optimizer,args.epochs, args.learning_rate, len(train_loader))

if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)

evaluator = create_detection_evaluator(args.model,
                                       model, 
                                       device, 
                                       coco_api_val_dataset)

trainer = create_detection_trainer(args.model, 
                                   model, 
                                   optimizer, 
                                   device,
                                   val_loader,
                                   evaluator,
                                   loss_fn = loss_fn,
                                   logging = local_rank == 0
                                   )

trainer.add_event_handler(
    Events.ITERATION_COMPLETED, scheduler,
)

if local_rank == 0:
    dirname = strftime("%d-%m-%Y_%Hh%Mm%Ss", localtime())
    dirname = "checkpoints/" +  args.model + "/{}".format(dirname)
    
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


