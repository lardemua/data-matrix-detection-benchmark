# data-matrix-detection-benchmark
Benchmark for the detection of data matrix landmarks using Deep Learning architectures.


# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Hardware](#hardware) 
- [Object Detection](#object-detection)
  * [Faster RCNN](#faster-rcnn)
    * [Resnet50FPN](#resnet50fpn)
    * [Resnet50](#resnet50)
    * [MobileNetV2](#mobilenetv2)
  * [SSD512](#ssd512)
  * [YOLOV3](#yolov3)


# Overview
This repo is an aggregation of Faster RCNN, SSD512 and YOLOV3(608) for Data Matrix detection. The performance comparison between all models is also performed. The training set images size from the original dataset were 8000*6000. The original training set images were reduced 4 times due to the faster forward propagation of the single shot methods and the way how Python deals with memory deallocation (see [Python Garbage Collection](https://www.digi.com/resources/documentation/digidocs/90001537/references/r_python_garbage_coll.htm)).   

# Hardware

Training and Evaluation: Nvidia RTX2080ti

# Object Detection

## Faster RCNN

### Resnet50FPN

To train:
````
python train_faster.py --batch_size 4 --learning_rate 1e-4 --epochs 50 --feature_extractor resnet50fpn
````

To evaluate:
````
python eval_faster.py --state_dict <PATH to model.pth> --feature_extractor resnet50fpn
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.608    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.780    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.713    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.242    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.562    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.756    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.301    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.658    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.658    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.267    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.609    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.806    |

framerate: 5fps

### Resnet50
To train:
````
python train_faster.py --batch_size 2 --learning_rate 1e-3 --epochs 50 --feature_extractor resnet50
````

To evaluate:
````
python eval_faster.py --state_dict <PATH to model.pth> --feature_extractor resnet50
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.475    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.750    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.518    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.115    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.403    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.620    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.241    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.549    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.554    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.286    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.479    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.694    |




### MobileNetV2

To train:
````
python train_faster.py --batch_size 4 --learning_rate 1e-4 --epochs 50
````

To evaluate:
````
python eval_faster.py --state_dict <PATH to model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.442    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.699    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.459    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.015    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.341    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.632    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.236    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.502    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.503    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.113    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.414    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.686    |

frame rate: 10 fps

## SSD512


To train:
````
python train_ssd.py --model ssd512 --batch_size 16 --learning_rate 1e-3 --weight_decay 4e-5 --epochs 150
````

To evaluate:
````
python eval_ssd.py --model ssd512 -state_dict <PATH to model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.216    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.368    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.218    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.065    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.292    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.159    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.297    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.297    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.067    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.405    |

framerate: 71fps

Evaluation(mosaic mode):

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.359    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.575    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.402    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.027    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.325    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.495    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.207    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.464    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.466    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.023    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.446    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.602    |

framerate:7fps


## YOLOV3

To train:
````
python train_yolov3.py --model yolov3 --batch_size 8 --weight_decay 4e-5 --eps 200
````

To evaluate:
````
python eval_yolov3.py --model yolov3 --state_dict <model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.007    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.034    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.002    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.001    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.015    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.026    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.035    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.035    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.003    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.066    |