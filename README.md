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
- [YOLO](#yolo)
    * [v3 SPP](#v3-spp)
    * [v3](#v3)
    * [v4](#v4)  
- [Acknowledgements](#acknowledgements)


# Overview
This repo is an aggregation of Faster RCNN, SSD512 and YOLOV3(608) for Data Matrix detection. The performance comparison between all models is also performed. The training set images size from the original dataset were 8000*6000. The original training set images were reduced 4 times due to the faster forward propagation of the single shot methods and the way how Python deals with memory deallocation (see [Python Garbage Collection](https://www.digi.com/resources/documentation/digidocs/90001537/references/r_python_garbage_coll.htm)).   

# Hardware

Training and Evaluation: Nvidia RTX2080ti


# Faster RCNN

## Resnet50FPN

To train:
````
python scripts/train_faster.py --batch_size 4 --learning_rate 2.5e-4 --epochs 50 --feature_extractor resnet50fpn
````

To evaluate:
````
python scripts/eval_faster.py --state_dict <PATH to model.pth> --feature_extractor resnet50fpn
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.681    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.924    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.840    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.480    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.676    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.737    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.311    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.752    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.725    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.642    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.735    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.798    |

framerate: 5fps

## Resnet50
To train:
````
python scripts/train_faster.py --batch_size 2 --learning_rate 2.5e-4 --epochs 50 --feature_extractor resnet50
````
     
To evaluate:
````
python scripts/eval_faster.py --state_dict <PATH to model.pth> --feature_extractor resnet50
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.543    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.795    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.595    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.186    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.455    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.707    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.277    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.610    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.611    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.316    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.524    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.768    |

framerate: 5.3fps


## MobileNetV2

To train:
````
python scripts/train_faster.py --batch_size 4 --learning_rate 2.5e-4 --epochs 50 --feature_extractor mobilenetv2
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
python scripts/train_ssd.py --model ssd512 --batch_size 16 --learning_rate 1e-3 --weight_decay 4e-5 --epochs 300 --feature_extractor mobilenetv2
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


# YOLO 

## v3 SPP

Input size: width = 1024

To train:
````
python scripts/train_yolo.py --model yolov3_spp --batch_size 4 --epochs 400
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov3_spp --batch_size 4 --state_dict <model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.340    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.590    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.344    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.241    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.524    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.291    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.188    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.458    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.477    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.384    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.622    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.650    |

  framerate: 43.5 fps

## v3 

Input size: width = 1024

To train:
````
python scripts/train_yolo.py --model yolov3 --batch_size 4 --epochs 400 
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov3 --batch_size 4 --state_dict <model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.305    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.572    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.293    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.178    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.496    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.376    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.179    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.405    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.427    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.321    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.593    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.600    |

  framerate: 40 fps

## v4

Input size: width = 1024

To train:
````
python scripts/train_yolo.py --model yolov4 --batch_size 2 --epochs 400 
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov4 --batch_size 2 --state_dict <model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.196    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.358    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.191    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.058    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.411    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.487    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.129    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.281    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.312    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.179    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.517    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.725    |

  framerate: 33 fps

# Acknowledgements

Repos: [MobileNetV2 + Single Shot Multibox Detector](https://github.com/qfgaohao/pytorch-ssd) and [YOLO](https://github.com/ultralytics/yolov3)

Project: This work was supported by the PRODUTECH II SIF-POCI-01-0247-FEDER-024541 Project.