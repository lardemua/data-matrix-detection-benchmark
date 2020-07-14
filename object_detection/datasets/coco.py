import torch
import cv2 
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pickle

PATH_IMG_INFO_TRAIN = "/srv/datasets/coco/annotations/train_imgs.txt"
PATH_ANNS_INFO_TRAIN = "/srv/datasets/coco/annotations/anns_info_train.json"
PATH_IMAGES_TRAIN = "/srv/datasets/coco/images/train2017"

PATH_IMG_INFO_VAL = "/srv/datasets/coco/annotations/test_imgs.txt"
PATH_ANNS_INFO_VAL = "/srv/datasets/coco/annotations/anns_info_val.json"
PATH_IMAGES_VAL = "/srv/datasets/coco/images/val2017"


class COCODetection(object):
    def __init__(self, transforms = None, target_transform = None, mode = "train"):
        self.transforms = transforms 
        self.target_transform = target_transform
        self.mode = mode
        
        if self.mode == "train" or self.mode == "val":
            if self.mode=="train":
                img_info_file = PATH_IMG_INFO_TRAIN
                anns_info_file = PATH_ANNS_INFO_TRAIN
                imgs_path = PATH_IMAGES_TRAIN
            else:
                img_info_file = PATH_IMG_INFO_VAL
                anns_info_file = PATH_ANNS_INFO_VAL
                imgs_path = PATH_IMAGES_VAL

            with open(img_info_file, "rb") as fp:   # Unpickling
                imgs_data = pickle.load(fp)
            with open(anns_info_file) as json_file:
                anns_data = json.load(json_file)
                
            self.imgs_data = imgs_data
            self.anns_data = anns_data
            self.imgs_path = imgs_path

        else:
            raise Exception("Oops. There are only two modes: train and val!")
        
    
    def __getitem__(self, idx):
        img_info = self.imgs_data[idx]
        filename = img_info["filename"]
        img_id = img_info["id"]
        
        img = cv2.imread(os.path.join(self.imgs_path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width, height = img.shape[1], img.shape[0]
        img = np.array(img)
        
        ann_boxes = self.anns_data[str(img_id)]["boxes"] #[x, y, w, h]
        labels = self.anns_data[str(img_id)]["labels"]
        areas = self.anns_data[str(img_id)]["area"]
        iscrowd = self.anns_data[str(img_id)]["iscrowd"]
        
        boxes_xywh = np.array(ann_boxes)
        # boxes to [xmin, y_min, x_max, y_max]
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:,0] = boxes_xywh[:,0]
        boxes_xyxy[:,1] = boxes_xywh[:,1]
        boxes_xyxy[:,2] = boxes_xywh[:,0] +  boxes_xywh[:,2]
        boxes_xyxy[:,3] = boxes_xywh[:,1] + boxes_xywh[:,3]
        
        boxes_ssd = np.array(boxes_xyxy, dtype = np.float32)
        labels_ssd = np.array(labels, dtype = np.int64)
        boxes = torch.as_tensor(boxes_xyxy, dtype = torch.float32)
        labels = torch.as_tensor(labels, dtype = torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor(img_id)
        target["area"] = torch.as_tensor(areas,dtype = torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype = torch.int64)
        
        sample = {
            "image":img,
            "bboxes":boxes,
            "labels":labels
        }
        
        if (self.transforms is not None) and (self.target_transform is None):
            augmented = self.transforms(**sample)
            img = augmented['image']
            target['boxes'] = torch.as_tensor(augmented['bboxes'],  dtype = torch.float32)
            target['labels'] = torch.as_tensor(augmented['labels'], dtype = torch.int64)
            
        elif (self.transforms is not None) and (self.target_transform is not None): #SSD
            img, boxes, labels = self.transforms(img, boxes_ssd, labels_ssd)
            target['boxes'] = boxes
            target['labels'] = labels
            boxes, labels = self.target_transform(target['boxes'], target['labels'])
            return img, boxes, labels 
        
        return img, target
        
    def __len__(self):
        return len(self.imgs_data)
    
