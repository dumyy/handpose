# This is a realtime demo for CrossInfoNet

**The original demo is from [deepprior++](https://github.com/moberweger/deep-prior-pp/tree/master/src)**



## realsense-lib install

You can use pip to install pyrealsense2 esaily.

    sudo pip install pyrealsense2
  
Or you can also build the python lib from [source](https://github.com/dumyy/librealsense/tree/master/wrappers/python) 

In this demo, we use Realsense SR300 device.

## realtime running

Just `cd realtime_demo` and `run` the command in you Terminal:

    python real_time_show_demo.py --dataset ${dataset_name}

eg. You want to load nyu model for realtime testing, use `nyu` to replace `${dataset_name}`,just like follow:

    python real_time_show_demo.py --dataset nyu
    python real_time_show_demo.py --dataset msra

I have changed the model name from `crossInfoNet_nyu.ckpt` to `crossInfoNet_NYU.ckpt`.
 
## Something else

When you are testing, 30cm -40 cm is a good choice for your **right hand** to the camera.
 
For the **icvl** demo, the depth image has been flipped.

The end GUI screenshot is like the follow show.

|model|screenshot1|screenshot2|
|:---:|:---:|:---:|
|msra|![shot0](https://github.com/dumyy/handpose/blob/master/figs/Screenshot%202019-06-06%2022:44:15.png)|![b0](https://github.com/dumyy/handpose/blob/master/figs/Screenshot%202019-06-06%2022:44:31.png)
|nyu|![shot1](https://github.com/dumyy/handpose/blob/master/figs/Screenshot%202019-06-07%2019:59:10.png)|![b1](https://github.com/dumyy/handpose/blob/master/figs/Screenshot%202019-06-07%2020:09:37.png)
|bighand|![shot2](https://github.com/dumyy/handpose/blob/master/figs/Screenshot%202019-06-07%2020:11:52.png)|![b2](https://github.com/dumyy/handpose/blob/master/figs/Screenshot%202019-06-07%2020:12:36.png)
|icvl|![shot3](https://github.com/dumyy/handpose/blob/master/figs/Screenshot%202019-06-07%2023:42:14.png)|![b3](https://github.com/dumyy/handpose/blob/master/figs/Screenshot%202019-06-07%2023:44:36.png)





