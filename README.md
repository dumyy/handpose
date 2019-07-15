# CrossInfoNet: Multi-Task Information Sharing Based Hand Pose Estimation

**This respository contains the implementation details of this [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Du_CrossInfoNet_Multi-Task_Information_Sharing_Based_Hand_Pose_Estimation_CVPR_2019_paper.pdf)

The [project](https://sites.google.com/view/dumyy/home/cvpr2019) page can be found here.


## Requirments

- python 2.7
- tensorflow == 1.3~1.9
- matplotlib < 3.0
- numpy
- scipy
- pillow
- some other packages important

our code is tested in Ubuntu 14.04 and 16.04 environment with GTX 1080 and RTX 2080 TI.

2 examples

- config 1: gtx1080+cuda9.0+cudnn7.x+tensorflow1.9+ubuntu16.04
- config 2: rtx2080ti+cuda10+cudnn7.x+tensorflow1.13+ubuntu16.04

You should match right cudnn version in this [site](https://developer.nvidia.com/rdp/cudnn-archive)

## Data Reprocessing

Download the datasets (ICVL, NYU, and MSRA).

Thanks [DeepPrior++](https://arxiv.org/pdf/1708.08325.pdf) for providing the base data reprocess and online data augmentation codes.

We use the precomputed centers of [V2V-PoseNet](http://openaccess.thecvf.com/content_cvpr_2018/papers/Moon_V2V-PoseNet_Voxel-to-Voxel_Prediction_CVPR_2018_paper.pdf)@[mks0601](https://github.com/mks0601/V2V-PoseNet_RELEASE)
when training ICVL and NYU datasets. 

Please refer to `cache/${dataset-name}/readme.md` for more details.

## Traing and Testing

Here we provide an example for NYU training. 

    cd $ROOT
    cd network/NYU
    python train_and_test.py

Here `$ROOT` is the root path that you put this project.

For testing, just run the command in the path `$ROOT/network/NYU/`

    python test_nyu_cross.py

For the MSRA dataset, just cd `$ROOT/network/MSRA/` directory, then run the train or test file, as follow:

    train:  python train_and_test.py --test-sub ${sub-num}
    test:   python test_msra.py --test-sub ${sub-num}

`${sub-num}` is the subject that you use to test while cross-validation.

In the end, you can use `python combtxt.py` to combine the 9 test results.
    

## Results

When testing, the model outputs the mean joint error. If you want to show the qualitative results, just let the `visual=True`.
We use [awesome-hand-pose-estimation](https://github.com/xinghaochen/awesome-hand-pose-estimation)
to evaluate the accuracy of the proposed *CrossInfoNet* on the ICVL, NYU and MSRA datasets. The predicted labels are [here](https://github.com/dumyy/handpose/tree/master/results/).

We also tested the perfomance on the HANDS 17 frame-based hand pose estiamtion challenge dataset. Here is the result on Feb.2, 2019.

![hands](https://github.com/dumyy/handpose/blob/master/figs/hands.png)


## Realtime demo

More details can be found in the [`realtime_demo`](https://github.com/dumyy/handpose/tree/master/realtime_demo) directory.

