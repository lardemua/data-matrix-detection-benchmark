# data-matrix-detection-benchmark
Benchmark for the detection of data matrix landmarks using Deep Learning architectures.


# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Hardware](#hardware) 
- [Faster RCNN](#faster-rcnn)
    * [Resnet50FPN](#resnet50fpn)
    * [Resnet50](#resnet50)
    * [MobileNetV2](#mobilenetv2)
- [SSD512](#ssd512)
    * [Resnet50](#resnet50)
    * [MobileNetV2](#mobilenetv2)
- [YOLOV3](#yolov3)
- [Acknowledgements](#acknowledgements)


# Overview
This repo is an aggregation of Faster RCNN, SSD512 and YOLOV3(608) for Data Matrix detection. The performance comparison between all models is also performed. The training set images size from the original dataset were 8000*6000. The original training set images were reduced 4 times due to the faster forward propagation of the single shot methods and the way how Python deals with memory deallocation (see [Python Garbage Collection](https://www.digi.com/resources/documentation/digidocs/90001537/references/r_python_garbage_coll.htm)).   

# Hardware

Training and Evaluation: Nvidia RTX2080ti


# Faster RCNN

## Resnet50FPN

To train:
````
python scripts/train_faster.py --batch_size 4 --learning_rate 1e-4 --epochs 50 --feature_extractor resnet50fpn
````

To evaluate:
````
python scripts/eval_faster.py --state_dict <PATH to model.pth> --feature_extractor resnet50fpn
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

## Resnet50
To train:
````
python scripts/train_faster.py --batch_size 2 --learning_rate 1e-3 --epochs 50 --feature_extractor resnet50
````

To evaluate:
````
python scripts/eval_faster.py --state_dict <PATH to model.pth> --feature_extractor resnet50
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

framerate: 5.3fps


## MobileNetV2

To train:
````
python scripts/train_faster.py --batch_size 4 --learning_rate 1e-4 --epochs 50 --feature_extractor mobilenetv2
````

To evaluate:
````
python scripts/eval_faster.py --feature_extractor mobilenetv2 --state_dict <PATH to model.pth> 
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

# SSD512

## Resnet50

To train:
````
python scripts/train_ssd.py --model ssd512 --batch_size 16 --learning_rate 1e-3 --weight_decay 4e-5 --epochs 300 --feature_extractor resnet50
````

To evaluate:
````
python scripts/eval_ssd.py --model ssd512 --feature_extractor resnet50 --state_dict <PATH to model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.386    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.587    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.446    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.241    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.539    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.207    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.440    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.440    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.262    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.614    |

framerate: 58 fps


## MobileNetV2

To train:
````
python scripts/train_ssd.py --model ssd512 --batch_size 16 --learning_rate 1e-3 --weight_decay 4e-5 --epochs 300
````

To evaluate:
````
python scripts/eval_ssd.py --model ssd512 --feature_extractor mobilenetv2 --state_dict <PATH to model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.246    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.422    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.257    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.079    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.329    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.162    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.333    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.333    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.080    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.449    |

framerate: 71fps


# YOLOV3

To train:
````
python scripts/train_yolov3.py --model yolov3 --batch_size 8 --weight_decay 4e-5 --epochs 300
````

To evaluate:
````
python scripts/eval_yolov3.py --model yolov3 --state_dict <model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.       |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.       |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.       |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.       |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.       |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.       |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.       |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.       |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.       |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.       |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.       |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.       |

  framerate: 


# Acknowledgements

Repos: [MobileNetV2 + Single Shot Multibox Detector](#https://github.com/qfgaohao/pytorch-ssd)
Project: This work was supported by the PRODUTECH II SIF-POCI-01-0247-FEDER-024541 Project.