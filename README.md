# CrossInfoNet: Multi-Task Information Sharing Based Hand Pose Estimation

**This respository contains the implementation details of this [paper]().**

## Requirments

- python 2.7
- tensorflow == 1.3~1.9
- matplotlib < 3.0
- numpy
- scipy
- pillow
- some other packages important

our code is tested in Ubuntu 16.04 and 16.04 enviroment with GTX 1080 and RTX 2080 TI

## Data Reprocessing

Download the datasets (ICVL, NYU, and MSRA).

Thanks [DeepPrior++](https://arxiv.org/pdf/1708.08325.pdf) for providing the base data reprocess and online data augmentation codes.

We use the precomputed centers of [V2V-PoseNet](http://openaccess.thecvf.com/content_cvpr_2018/html/Moon_V2V-PoseNet_Voxel-to-Voxel_Prediction_CVPR_2018_paper.html)@[mks0601](https://github.com/mks0601/V2V-PoseNet_RELEASE)
when training ICVL and NYU datasets. 

## Traing and Testing

Here we provide an example for NYU training. 

    cd $ROOT
    cd network/NYU
    python train_and_test.py

Here `$ROOT` is the root path that you put this project.

For testing, just run the command in the path `$ROOT/network/NYU/`

    python test_nyu_cross.py

## Results

When testing, the model outputs the mean joint error. If you want to show the qualitative results, just let the `visual=True`.
We use [awesome-hand-pose-estimation](https://github.com/xinghaochen/awesome-hand-pose-estimation)
to evaluate the accuracy of the proposed *CrossInfoNet* on the ICVL, NYU and MSRA datasets. The predicted labels are [here](https://github.com/dumyy/handpose/tree/master/results/).

We also tested the perfomance on the HANDS 17 frame-based hand pose estiamtion challenge dataset. Here is the result on Feb.2, 2019.

![hands](https://github.com/dumyy/Projects/blob/master/figs/result/hands.png)

