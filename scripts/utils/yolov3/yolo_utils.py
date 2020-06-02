from __future__ import division
import os
import time
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

# From: https://github.com/zhanghanduo/yolo3_pytorch/blob/0e8943ef9047c63bec25082c09dbbb29b4a133d0/common/utils.py#L14

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_all_boxes(output, conf_thresh, num_classes, only_objectness=1, validation=False):
    # total number of inputs (batch size)
    # first element (x) for first tuple (x, anchor_mask, num_anchor)
    tot = output[0]['x'].data.size(0)
    all_boxes = [[] for i in range(tot)]
    for i in range(len(output)):
        pred, anchors, num_anchors = output[i]['x'].data, output[i]['a'], output[i]['n'].item()
        b = get_region_boxes(pred, conf_thresh, num_classes, anchors, num_anchors, \
                             only_objectness=only_objectness, validation=validation)
        for t in range(tot):
            all_boxes[t] += b[t]
    return all_boxes


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    device = torch.device("cuda")
    anchors = anchors.to(device)
    anchor_step = anchors.size(0) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)
    cls_anchor_dim = batch * num_anchors * h * w

    t0 = time.time()
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes,
                                                                                                        cls_anchor_dim)

    grid_x = torch.linspace(0, w - 1, w).repeat(batch * num_anchors, h, 1).view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(cls_anchor_dim).to(
        device)
    ix = torch.LongTensor(range(0, 2)).to(device)
    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, ix[0]).repeat(1, batch, h * w).view(
        cls_anchor_dim)
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, ix[1]).repeat(1, batch, h * w).view(
        cls_anchor_dim)

    xs, ys = torch.sigmoid(output[0]) + grid_x, torch.sigmoid(output[1]) + grid_y
    ws, hs = torch.exp(output[2]) * anchor_w.detach(), torch.exp(output[3]) * anchor_h.detach()
    det_confs = torch.sigmoid(output[4])

    # by ysyun, dim=1 means input is 2D or even dimension else dim=0
    cls_confs = torch.nn.Softmax(dim=1)(output[5:5 + num_classes].transpose(0, 1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs, ys = convert2cpu(xs), convert2cpu(ys)
    ws, hs = convert2cpu(ws), convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))

    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def plot_boxes(img, boxes, savename=None, class_names=None, net_w=416, net_h=416):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    print("%d box(es) is(are) found" % len(boxes[0]))
    for i in range(len(boxes[0])):
        box = boxes[0][i]
        x1 = box[0]/net_w * width
        y1 = box[1]/net_h * height
        x2 = box[2]/net_w * width
        y2 = box[3]/net_h * height

        fnt = ImageFont.truetype("arial.ttf", 30)

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = int(box[6])
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], font=fnt, fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size // 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r', encoding='utf8') as fp:
        lines = fp.readlines()
    for line in lines:
        class_names.append(line.strip())
    return class_names


def image2torch(img):
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)
    return img


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs, self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        if targets is not None:
            #  build target
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors,
                                                                            in_w, in_h,
                                                                            self.ignore_threshold)
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()

            n_mask = torch.sum(mask)
            n_noobj_mask = torch.sum(noobj_mask)

            #  losses.
            loss_x = self.bce_loss(x * mask, tx * mask) #/ n_mask
            loss_y = self.bce_loss(y * mask, ty * mask) #/ n_mask
            loss_w = self.mse_loss(w * mask, tw * mask) #/ n_mask
            loss_h = self.mse_loss(h * mask, th * mask) #/ n_mask
            loss_conf = self.bce_loss(conf * mask, mask) + \
                        0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
            #loss_conf = self.bce_loss(conf * mask, 1.0 * mask) / n_mask + \
            #            0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0) / n_noobj_mask
            #loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1]) / n_mask
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                   loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                   loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
                   loss_h.item(), loss_conf.item(), loss_cls.item()
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # Calculate offsets for each grid
            grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
            # Calculate anchor w, h
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(bs):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                # Convert to position relative to box
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
        
                gt_box = torch.FloatTensor(np.array([0, 0,gw.item(), gh.item()])).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                # Masks
                if (gj < in_h) and (gi < in_w):
                    mask[b, best_n, gj, gi] = 1
                    # Coordinates
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    # Width and height
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
                    # object
                    tconf[b, best_n, gj, gi] = 1
                    # One-hot encoding of label
                    tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
    
    
def get_optimizer(config, net):
    optimizer = None
    params = None
    
    base_params = list(map(id, net.backbone.parameters()))
    logit_params = filter(lambda p: id(p) not in base_params, net.parameters())
    
    if not config["hyper_params"]["freeze_backbone"]:
        params = [
            {
                "params":logit_params, "lr":config["hyper_params"]["base_lr"]
            },
            {
                "params": net.backbone.parameters(), "lr":config["hyper_params"]["backbone_lr"]
            },
        ]
    else:
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {
                "params":logit_params, "lr": config["hyper_params"]["backbone_lr"]
            },
        ]
    optimizer = optim.SGD(params, momentum = 0.9,
                          weight_decay = config["optimizer"]["weight_decay"],
                          nesterov=(config["optimizer"]["type"] == "nesterov"))
    
    return optimizer






class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)


class ToTensor(object):
    def __init__(self, max_objects=50, is_debug=False):
        self.max_objects = max_objects
        self.is_debug = is_debug

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        if self.is_debug == False:
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)

        filled_labels = np.zeros((self.max_objects, 5), np.float32)
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(filled_labels)}

class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        image_new = np.pad(image, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = image_new.shape

        # Extract coordinates for unpadded + unscaled image
        x1 = w * (label[:, 1] - label[:, 3]/2)
        y1 = h * (label[:, 2] - label[:, 4]/2)
        x2 = w * (label[:, 1] + label[:, 3]/2)
        y2 = h * (label[:, 2] + label[:, 4]/2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        label[:, 1] = ((x1 + x2) / 2) / padded_w
        label[:, 2] = ((y1 + y2) / 2) / padded_h
        label[:, 3] *= w / padded_w
        label[:, 4] *= h / padded_h

        return {'image': image_new, 'label': label}

class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = tuple(new_size) #  (w, h)
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, self.new_size, interpolation=self.interpolation)
        return {'image': image, 'label': label}

class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                # Add gaussian noise to some images.
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # Add a value of -5 to 5 to each pixel.
                sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                # Change brightness of images (80-120% of original value).
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                # Improve or worsen the contrast of images.
                sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def __call__(self, sample):
        seq_det = self.seq.to_deterministic()
        image, label = sample['image'], sample['label']
        image = seq_det.augment_images([image])[0]
        return {'image': image, 'label': label}



