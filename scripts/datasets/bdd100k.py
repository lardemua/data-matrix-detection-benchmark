import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
import albumentations

PATH = '../../../../../../../srv/datasets/bdd100k/'
TRAINING_SIZE = '100k'
PATH_IMAGES = os.path.join(PATH, "images/" + TRAINING_SIZE)
PATH_LABELS = os.path.join(PATH, "labels")

LBLS_MAP = {
    0 : 'BACKGROUND', 
    1 : 'bus',
    2 : 'traffic light',
    3 : 'traffic sign',
    4 : 'person',
    5 : 'bike',
    6 : 'truck',
    7 : 'motor',
    8 : 'car',
    9 : 'train',
    10 : 'rider'    
}

class BDD100kDataset(object):
    """BDD100K dataset for object detection 
    Keyword arguments:
    - transforms: transformations to be applied based on
    - target transforms: for the SSD model
    albumentations library
    - mode (train or val)     
    """
    def __init__(self, transforms = None, target_transform = None, mode = 'train'):
        self.transforms = transforms
        self.target_transform = target_transform
        self.mode = mode
        if self.mode == 'train' or self.mode == 'val':
            self.imgs = os.listdir(os.path.join(PATH_IMAGES,self.mode))
            self.lbls = os.path.join(PATH_LABELS, 'bdd100k_labels_images_' + self.mode + '.json')
            self.labels_file = pd.read_json(self.lbls)
        else:
            raise Exception("Oops. There are only two modes: 'training' and 'validation'!")
            
    def __getitem__(self, idx):
        filename = self.labels_file["name"][idx]
        img = Image.open(os.path.join(PATH_IMAGES,self.mode, filename)).convert('RGB')
        width, height = img.size
        img = np.array(img)
        lbl_file_row = self.labels_file[self.labels_file.index == idx ]
        num_objs = len(lbl_file_row['labels'][idx])
        boxes = []
        obj_labels = []
        for i in range(num_objs):
            object_ =  lbl_file_row['labels'][idx][i]
            if 'box2d' in object_:
                coordinates = object_['box2d']
            else:
                continue
            boxes.append([coordinates['x1'], coordinates['y1'],coordinates['x2'],coordinates['y2']])
            lbl_mapped = list(LBLS_MAP.keys())[list(LBLS_MAP.values()).index(object_['category'])]
            obj_labels.append(lbl_mapped)
        boxes_ssd = np.array(boxes, dtype = np.float32)
        labels_ssd = np.array(obj_labels, dtype = np.int64)
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.tensor(obj_labels, dtype = torch.int64)
        image_id = torch.tensor(idx)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(boxes.shape[0], dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        sample = {'image':img, 
                  'bboxes':boxes,
                  'labels':labels
                 }
        if (self.transforms is not None) and (self.target_transform is None):
            augmented = self.transforms(**sample)
            img = augmented['image']
            target['boxes'] = torch.as_tensor(augmented['bboxes'],  dtype = torch.float32)
            target['labels'] = torch.as_tensor(augmented['labels'], dtype = torch.int64)
            
        elif (self.transforms is not None and self.target_transform is not None): #SSD 
            img, boxes, labels = self.transforms(img, boxes_ssd, labels_ssd)
            target['boxes'] = boxes
            target['labels'] = labels
            boxes, labels = self.target_transform(target['boxes'], target['labels'])
            return img, boxes, labels 
        
        return img, target

    def __len__(self):
        return self.labels_file.shape[0]-1

