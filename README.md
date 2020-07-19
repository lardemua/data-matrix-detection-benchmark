# data-matrix-detection-benchmark
Benchmark for the detection of data matrix landmarks using Deep Learning architectures.


# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Hardware](#hardware) 
- [Faster RCNN](#faster-rcnn)
    * [Resnet50FPN pretrained](#resnet50fpn-pretrained)
    * [Resnet50FPN from scratch](#resnet50fpn-from-scratch)
    * [Resnet50 pretrained](#resnet50-pretrained)
    * [Resnet50 from scratch](#resnet50-from-scratch)
    * [MobileNetV2 pretrained](#mobilenetv2-pretrained)
    * [MobileNetV2 from scratch](#mobilenetv2-from-scratch)
- [SSD512](#ssd512)
    * [Resnet50](#resnet50)
    * [MobileNetV2](#mobilenetv2)
- [YOLO](#yolo)
    * [v3](#v3)
    * [v3 mosaic aug](#v3-mosaic-aug)
    * [v3 SPP](#v3-spp)
    * [v3 SPP mosaic aug](#v3-spp-mosaic-aug)
    * [v4](#v4)
    * [v4 mosaic aug](#v4-mosaic-aug)   
- [Acknowledgements](#acknowledgements)


# Overview
This repo is a comparison between Faster RCNN, SSD512 and YOLOV3(608) for Data Matrix detection. The performance comparison between all models is also performed. The training set images size from the original dataset were 8000*6000. The original training set images were reduced 4 times due to the faster forward propagation of the single shot methods and the way how Python deals with memory deallocation (see [Python Garbage Collection](https://www.digi.com/resources/documentation/digidocs/90001537/references/r_python_garbage_coll.htm)).   

# Hardware

Training and Evaluation: Nvidia RTX2080ti


# Faster RCNN 

**Weights:**

* Resnet50 FPN: COCO
* Resnet50 and MobileNetV2: Imagenet

## Resnet50FPN pretrained

To train:
````
python scripts/train_faster.py --batch_size 4 --learning_rate 2.5e-4 --epochs 10 --feature_extractor resnet50fpn
````

To evaluate:
````
python scripts/eval_faster.py --feature_extractor resnet50fpn --state_dict <PATH to model.pth> 
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.681    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.924**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.840    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.480    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.676    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.737    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.311    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.752    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.725**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.642    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.735    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.798    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.728    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.953**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.905    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.210    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.710    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.809    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.140    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.740    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.772**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.452    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.754    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.844    |



## Resnet50FPN from scratch

To train:
````
python scripts/train_faster.py --batch_size 4 --learning_rate 2.5e-4 --epochs 30 --feature_extractor resnet50fpn --pretrained False
````

To evaluate:
````
python scripts/eval_faster.py --feature_extractor resnet50fpn --state_dict <PATH to model.pth> 
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.599    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.780**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.716    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.337    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.529    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.736    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.280    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.648    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.648**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.386    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.568    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.790    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.722    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.906**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.854    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.222    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.698    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.815    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.145    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.735    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.753**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.396    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.730    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.851    |

framerate: 5.4fps




## Resnet50 pretrained
To train:
````
python scripts/train_faster.py --batch_size 2 --learning_rate 2.5e-4 --epochs 30 --feature_extractor resnet50
````
     
To evaluate:
````
python scripts/eval_faster.py --feature_extractor resnet50 --state_dict <PATH to model.pth> 
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.543    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.795**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.595    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.186    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.455    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.707    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.277    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.610    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.611**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.316    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.524    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.768    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95    |     all   |    100    |     0.652    |
  | **Average Precision  (AP)**  |   **0.50**       |   **all** |  **100**  |   **0.894**  |
  | Average Precision  (AP)      |     0.75         |     all   |    100    |     0.750    |
  | Average Precision  (AP)      |     0.50:0.95    |   small   |    100    |     0.032    |
  | Average Precision  (AP)      |     0.50:0.95    |  medium   |    100    |     0.603    |
  | Average Precision  (AP)      |     0.50:0.95    |   large   |    100    |     0.806    |
  | Average Recall     (AR)      |     0.50:0.95    |     all   |      1    |     0.143    |
  | Average Recall     (AR)      |     0.50:0.95    |     all   |     10    |     0.665    |
  | **Average Recall     (AR)**  |   **0.50:0.95**  |   **all** |  **100**  |   **0.688**  |
  | Average Recall     (AR)      |     0.50:0.95    |   small   |    100    |     0.145    |
  | Average Recall     (AR)      |     0.50:0.95    |  medium   |    100    |     0.644    |
  | Average Recall     (AR)      |     0.50:0.95    |   large   |    100    |     0.840    |



## Resnet50 from scratch
To train:
````
python scripts/train_faster.py --batch_size 2 --learning_rate 1e-3 --epochs 200 --feature_extractor resnet50 --pretrained False
````
     
To evaluate:
````
python scripts/eval_faster.py --feature_extractor resnet50 --state_dict <PATH to model.pth>
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.499    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.724**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.547    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.166    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.412    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.661    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.255    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.555    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.555**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.255    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.463    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.718    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.618    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.855**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.730    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.033    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.548    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.794    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.141    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.640    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.661**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.112    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.614    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.824    |

framerate: 3.9


## MobileNetV2 pretrained

To train:
````
python scripts/train_faster.py --batch_size 4 --learning_rate 2.5e-4 --epochs 30 --feature_extractor mobilenetv2
````

To evaluate:
````
python scripts/eval_faster.py --feature_extractor mobilenetv2 --state_dict <PATH to model.pth> 
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.458    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.745**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.468    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.023    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.381    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.624    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.245    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.519    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.519**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.152    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.461    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.668    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.549    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.857**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.595    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.004    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.462    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.767    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.137    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.564    |
 | **Average Recall     (AR)**   |   **0.50:0.95** |   **all**  |  **100**  |   **0.586**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.048    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.511    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.804    |



## MobileNetV2 from scratch

To train:
````
python scripts/train_faster.py --batch_size 4 --learning_rate 1e-3 --epochs 200 --feature_extractor mobilenetv2 --pretrained False
````

To evaluate:
````
python scripts/eval_faster.py --feature_extractor mobilenetv2 --state_dict <PATH to model.pth> 
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.394    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.619**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.420    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.055    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.285    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.578    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.215    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.449    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.449**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.077    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.350    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.636    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.522    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.783**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.583    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.432    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.743    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.137    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.552    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.566**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.009    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.494    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.780    |

frame rate: 7.7 fps



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

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.386    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.587**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.446    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.241    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.539    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.207    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.440    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.440**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.262    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.614    |


On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.455    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.683**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.546    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.377    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.649    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.128    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.490    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.490**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.419    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.691    |

framerate: 48.3 fps


## MobileNetV2

To train:
````
python scripts/train_ssd.py --model ssd512 --batch_size 16 --learning_rate 1e-3 --weight_decay 4e-5 --epochs 300 --feature_extractor mobilenetv2
````

To evaluate:
````
python scripts/eval_ssd.py --model ssd512 --feature_extractor mobilenetv2 --state_dict <PATH to model.pth>
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.246    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.422**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.257    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.079    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.329    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.162    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.333    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.333**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.080    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.449    |


On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.255    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.441**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.270    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.128    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.529    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.108    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.277    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.277**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.000    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.151    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.569    |

framerate: 67fps




# YOLO 

Input size: width = 896

## v3 

To train:
````
python scripts/train_yolo.py --model yolov3 --batch_size 4 --epochs 400 
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov3 --batch_size 4 --state_dict <model.pth>
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.350    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.642**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.325    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.265    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.545    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.800    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.189    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.443    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.451**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.361    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.643    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.800    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.543    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.829**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.625    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.385    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.658    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.725    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.127    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.586    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.624**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.507    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.715    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.768    |


## v3 mosaic aug

To train:
````
python scripts/train_yolo.py --model yolov3 --batch_size 4 --epochs 400 --imgs_rect False
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov3 --batch_size 4 --state_dict <model.pth>
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.375    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.646**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.380    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.276    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.588    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.900    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.209    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.453    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.463**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.367    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.668    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.900    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.594    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.857**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.707    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.435    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.704    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.770    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.132    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.626    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.663**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.544    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.755    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.804    |

  framerate: 39.8 fps

## v3 SPP

To train:
````
python scripts/train_yolo.py --model yolov3_spp --batch_size 4 --epochs 400
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov3_spp --batch_size 4 --state_dict <model.pth>
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.364    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.665**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.366    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.267    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.566    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.800    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.198    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.459    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.467**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.378    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.658    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.800    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.568    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.851**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.659    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.406    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.687    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.755    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.132    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.607    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.649**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.526    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.744    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.800    |



## v3 SPP mosaic aug

To train:
````
python scripts/train_yolo.py --model yolov3_spp --batch_size 4 --epochs 400 --imgs_rect False
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov3_spp --batch_size 4 --state_dict <model.pth>
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.353    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.600**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.360    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.248    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.580    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.900    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.194    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.433    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.439**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.336    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.659    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.900    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.594    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.863**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.698    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.437    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.706    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.769    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.131    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.626    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.668**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.550    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.760    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.808    |
   
  framerate: 37.1 fps

## v4

To train:
````
python scripts/train_yolo.py --model yolov4 --batch_size 2 --epochs 400 
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov4 --batch_size 2 --state_dict <model.pth>
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.371    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.638**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.408    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.273    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.584    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.900    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.206    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.462    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.473**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.372    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.689    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.900    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.566    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.826**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.678    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.387    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.696    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.743    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.132    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.602    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.643**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.503    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.753    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.792    |
  
  framerate: 55 fps

## v4 mosaic aug

To train:
````
python scripts/train_yolo.py --model yolov4 --batch_size 2 --epochs 400 --imgs_rect False
````

To evaluate:
````
python scripts/eval_yolo.py --model yolov4 --batch_size 2 --state_dict <model.pth>
````

On validation set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.392    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.627**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.428    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.289    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.626    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.900    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.213    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.471    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.477**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.367    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.711    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.900    |

On test set:

  |       Metric                 |  IoU Thresholds |    Scales  |  maxDets  | AP/AR values |
  | :--------------------------: | :-------------: | :--------: | :-------: | :----------: |
  | Average Precision  (AP)      |     0.50:0.95   |     all    |    100    |     0.606    |
  | **Average Precision  (AP)**  |   **0.50**      |   **all**  |  **100**  |   **0.846**  |
  | Average Precision  (AP)      |     0.75        |     all    |    100    |     0.704    |
  | Average Precision  (AP)      |     0.50:0.95   |   small    |    100    |     0.427    |
  | Average Precision  (AP)      |     0.50:0.95   |  medium    |    100    |     0.729    |
  | Average Precision  (AP)      |     0.50:0.95   |   large    |    100    |     0.787    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |      1    |     0.137    |
  | Average Recall     (AR)      |     0.50:0.95   |     all    |     10    |     0.638    |
  | **Average Recall     (AR)**  |   **0.50:0.95** |   **all**  |  **100**  |   **0.680**  |
  | Average Recall     (AR)      |     0.50:0.95   |   small    |    100    |     0.552    |
  | Average Recall     (AR)      |     0.50:0.95   |  medium    |    100    |     0.780    |
  | Average Recall     (AR)      |     0.50:0.95   |   large    |    100    |     0.816    |

  framerate: 47.6 fps

# Acknowledgements

Repos: [MobileNetV2 + Single Shot Multibox Detector](https://github.com/qfgaohao/pytorch-ssd) and [Ultralytics](https://github.com/ultralytics/yolov3)

Project: This work was supported by the PRODUTECH II SIF-POCI-01-0247-FEDER-024541 Project.


