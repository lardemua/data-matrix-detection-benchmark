import albumentations as albu
from albumentations.pytorch import ToTensor

def get_tfms_faster():
  """Function that returns the transformations to
  be applied to both loaders (training, validation) 
  Keyword arguments:
  - cropsize: tupple
  - scalling: list of two values     
  """

  train_tfms = albu.Compose([
    albu.OneOf([
            albu.augmentations.transforms.RandomSizedBBoxSafeCrop(3024,4032,p=0.25),
            albu.augmentations.transforms.RandomSizedBBoxSafeCrop(1080,1920,p=0.25),
            albu.augmentations.transforms.Resize(600,800, p=0.5),
        ], p=1),
    albu.augmentations.transforms.MultiplicativeNoise(),
    albu.augmentations.transforms.RandomContrast(limit=0.5),
    #albu.augmentations.transforms.Resize(600,800),
    albu.HorizontalFlip(),
    ToTensor(),
  ],bbox_params = albu.BboxParams(format='pascal_voc', 
                                 min_area = 0., 
                                 min_visibility = 0., 
                                 label_fields=['labels']))
  val_tfms = albu.Compose([
    ToTensor(),
  ],bbox_params = albu.BboxParams(format='pascal_voc', 
                                 min_area = 0., 
                                 min_visibility = 0., 
                                 label_fields=['labels']))

  return train_tfms, val_tfms



def collate_fn(batch):
  """Function that organizes the batch
  Keyword arguments:
  - batch 
  """

  return list(zip(*batch))

def transform_inputs(images, targets, device):
  """Prepares data to pass forward the model
  during training 
  Keyword arguments:
  - images from batch
  - targets from batch
  - GPU device (e.g. "cuda:2")     
  """
  images = list(image.to(device) for image in images)
  #targets = [{k: v.to(device)  for k, v in t.items()} for t in targets]
  dev_targets = []
  target_dict = {}
  for t in targets:
    for k, v in t.items():
      if isinstance(v, list):
        target_dict.update({k:v})
      else:
        target_dict.update({k:v.to(device)})
    dev_targets.append(target_dict)
    
  return images, dev_targets