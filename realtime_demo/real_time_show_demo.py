#-*- coding:utf-8 -*-
#!usr/bin/python
import sys
sys.path.append('../')#add root directory
import numpy as np
import tensorflow as tf
import cv2
import pyrealsense2 as rs
from data.importers import DepthImporter
import argparse
from netlib.basemodel import basenet2

from util.realtimehandposepipeline import RealtimeHandposePipeline

import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers



class realsense_im(object):
    def __init__(self,image_size=(640,480)):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, image_size[0], image_size[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, image_size[0], image_size[1], rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

    def __get_depth_scale(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()

        depth_scale = depth_sensor.get_depth_scale()

        return depth_scale

    def get_image(self):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)
        color_image = np.asarray(color_frame.get_data(), dtype=np.uint8)
        color_image_pad = np.pad(color_image, ((20, 0), (0, 0), (0, 0)), "edge")
        depth_map_end = depth_image * self.__get_depth_scale() * 1000
        return depth_map_end,color_image_pad

    def process_end(self):
        self.pipeline.stop()

class model_setup():
    def __init__(self,dataset,model_path):
        self._dataset=dataset
        self.model_path=model_path
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, 96, 96, 1))

        self.hand_tensor=None
        self.model()
        self.saver = tf.train.Saver(max_to_keep=15)

    def __self_dict(self):
        if self._dataset=='icvl':
            return (16,6,10)
        if self._dataset=='nyu':
            return (14,9,5)
        if self._dataset in ['msra','bighand']:
            return (21,6,15)

    def __config(self):
        #set your own realsense info here
        flag=-1
        if self._dataset == 'icvl':
            flag=1
        di = DepthImporter(fx=475.268, fy=flag*475.268, ux=313.821, uy=246.075)
        config = None
        if self._dataset=='msra':
            config = {'fx': di.fx, 'fy': abs(di.fy), 'cube': (175, 175, 175), 'im_size': (96, 96)}
        if self._dataset == 'nyu':
            config = {'fx': di.fx, 'fy': abs(di.fy), 'cube': (250,250, 250), 'im_size': (96, 96)}
        if self._dataset == 'icvl':
            config = {'fx': di.fx, 'fy': abs(di.fy), 'cube': (240, 240, 240), 'im_size': (96, 96)}
        if self._dataset == 'bighand':
            config = {'fx': di.fx, 'fy': abs(di.fy), 'cube': (220, 220, 220), 'im_size': (96, 96)}
        return di, config

    def __crop_cube(self):
        return self.__config()[1]['cube'][0]
    def __joint_num(self):
        return self.__self_dict()[0]

    def model(self):
        outdims=self.__self_dict()
        fn = layers.l2_regularizer(1e-5)
        fn0 = tf.no_regularizer
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=fn,
                            biases_regularizer=fn0, normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm],
                                is_training=False,
                                updates_collections=None,
                                decay=0.9,
                                center=True,
                                scale=True,
                                epsilon=1e-5):
                pred_comb_ht, pred_comb_hand, pred_hand, pred_ht = basenet2(self.inputs, kp=1, is_training=False,
                                                                            outdims=outdims)

        self.hand_tensor=pred_hand

    def sess_run(self,realsense_dev):
        _di,_config=self.__config()

        rtp = RealtimeHandposePipeline(1, config=_config, di=_di, verbose=False, comrefNet=None)

        joint_num=self.__joint_num()
        cube_size=self.__crop_cube()

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.saver.restore(sess, self.model_path)
            while True:
                depth_frame, color_frame = realsense_dev.get_image()
                if self._dataset=='icvl':
                    depth_frame=np.fliplr(depth_frame)
                frame2 = depth_frame.copy()
                crop1, M, com3D = rtp.detect(frame2)
                crop = crop1.reshape(1, crop1.shape[0], crop1.shape[1], 1).astype('float32')
                pred_ = sess.run(self.hand_tensor, feed_dict={self.inputs: crop})

                norm_hand = np.reshape(pred_, (joint_num, 3))
                pose = norm_hand * cube_size / 2. + com3D

                img = rtp.show2(depth_frame, pose,self._dataset)
                img = rtp.addStatusBar(img)


                images1 = np.hstack((color_frame, img))
                cv2.imshow('frame', images1)
                cv2.imshow('crop', crop1)

                if cv2.waitKey(1) >= 0:
                    break
            realsense_dev.process_end()
        cv2.destroyAllWindows()



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='realsense_realtime_demo')
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()
    dataset_input=args.dataset

    realsense_dev=realsense_im(image_size=(640,480))
    if dataset_input == 'msra':
        #set the model from 0 to 8 that just you like.
        model = model_setup(dataset_input, '../model/crossInfoNet_{}3.ckpt'.format(dataset_input.upper()))
    else:
        model=model_setup(dataset_input,'../model/crossInfoNet_{}.ckpt'.format(dataset_input.upper()))

    model.sess_run(realsense_dev)














