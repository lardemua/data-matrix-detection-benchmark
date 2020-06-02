import os
import numpy as np
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils.yolov3.yolo_utils import Compose, ImageBaseAug, ResizeImage, ToTensor
from PIL import Image
import albumentations as albu

PATH = '../../../../../../../srv/datasets/bdd100k/'
TRAINING_SIZE = '100k'
PATH_IMAGES = os.path.join(PATH, 'images/' + TRAINING_SIZE)
PATH_LABELS = os.path.join(PATH, "labels")

LBLS_MAP = {
    0 : 'bus',
    1 : 'traffic light',
    2 : 'traffic sign',
    3 : 'person',
    4 : 'bike',
    5 : 'truck',
    6 : 'motor',
    7 : 'car',
    8 : 'train',
    9 : 'rider'    
}


class BDD100kDataset(Dataset):
    def __init__(self, img_size, mode = 'train',  is_debug=False):
        self.mode = mode
        self.img_size = img_size
        self.max_objects = 50
        self.is_debug = is_debug
        if self.mode == 'train' or self.mode == 'val':
            self.imgs = os.listdir(os.path.join(PATH_IMAGES,self.mode))
            self.lbls = os.path.join(PATH_LABELS, 'bdd100k_labels_images_' + self.mode + '.json')
            self.labels_file = pd.read_json(self.lbls)
        else:
            raise Exception("Oops. There are only two modes: 'training' and 'validation'!")
        self.transforms = Compose()
        if mode == 'train':
            self.transforms.add(ImageBaseAug())
        self.transforms.add(ResizeImage(self.img_size))
        self.transforms.add(ToTensor(self.max_objects, self.is_debug))
    def __getitem__(self, idx):
        filename = self.labels_file["name"][idx]
        img = cv2.imread(os.path.join(PATH_IMAGES,self.mode, filename), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
        width, height = img.shape[1], img.shape[0]
        img = np.array(img)
        lbl_file_row = self.labels_file[self.labels_file.index == idx ]
        num_objs = len(lbl_file_row['labels'][idx])
        target = []

        for i in range(num_objs):
            object_ =  lbl_file_row['labels'][idx][i]
            if 'box2d' in object_:
                coordinates = object_['box2d']
            else:
                continue
            lbl_mapped = list(LBLS_MAP.keys())[list(LBLS_MAP.values()).index(object_['category'])]
            normalized_bbox = albu.augmentations.bbox_utils.normalize_bbox(
                (coordinates['x1'], 
                 coordinates['y1'],
                 coordinates['x2'],
                 coordinates['y2']), 
                height, width)
            bbox_w = normalized_bbox[2] - normalized_bbox[0]
            bbox_h = normalized_bbox[3] - normalized_bbox[1]
            target.append(
                [lbl_mapped, 
                 normalized_bbox[0] + bbox_w / 2, 
                 normalized_bbox[1] + bbox_h / 2,
                 bbox_w,
                 bbox_h])
                          
        sample = {'image':img, 
                  'label':target
                 }
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample["image_path"] = os.path.join(PATH_IMAGES,self.mode, filename)
        sample["origin_size"] = str([width, height])
    
        return sample
        
        
    def __len__(self):
        return self.labels_file.shape[0]-1
        