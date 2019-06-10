"""Class for running the handpose estimation pipeline in realtime.

RealtimeHandposePipeline provides interface for running the pose estimation.
It is made of detection, image cropping and further pose estimation.

Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of DeepPrior.

DeepPrior is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""

from collections import deque
from multiprocessing import Process, Manager, Value
from ctypes import c_bool
import cv2
import time
import numpy
import copy
import numpy as np

from handdetector import HandDetector

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class RealtimeHandposePipeline(object):
    """
    Realtime pipeline for handpose estimation
    """

    # states of pipeline
    STATE_IDLE = 0
    STATE_INIT = 1
    STATE_RUN = 2

    # different hands
    HAND_LEFT = 0
    HAND_RIGHT = 1

    # different detectors
    DETECTOR_COM = 0

    def __init__(self, poseNet, config, di, verbose=False, comrefNet=None):
        """
        Initialize data
        :param poseNet: network for pose estimation
        :param config: configuration
        :param di: depth importer
        :param verbose: print additional info
        :param comrefNet: refinement network from center of mass detection
        :return: None
        """

        # handpose CNN
        self.importer = di
        self.poseNet = poseNet
        self.comrefNet = comrefNet
        # configuration
        self.initialconfig = copy.deepcopy(config)
        # synchronization between processes
        self.sync = Manager().dict(config=config, fid=0,
                                   crop=numpy.ones((96, 96), dtype='float32'),
                                   com3D=numpy.asarray([0, 0, 300], dtype='float32'),
                                   frame=numpy.ones((240, 320), dtype='float32'), M=numpy.eye(3))

        self.start_prod = Value(c_bool, False)
        self.start_con = Value(c_bool, False)
        self.stop = Value(c_bool, False)
        # for calculating FPS
        self.lastshow = time.time()
        self.runningavg_fps = deque(100*[0], 100)
        self.verbose = verbose
        # hand left/right
        self.hand = Value('i', self.HAND_LEFT)
        # initial state
        self.state = Value('i', self.STATE_IDLE)
        # detector
        self.detection = Value('i', self.DETECTOR_COM)
        # hand size estimation
        self.handsizes = []
        self.numinitframes = 50
        # hand tracking or detection
        self.tracking = Value(c_bool, False)
        self.lastcom = (0, 0, 0)
        # show different results
        self.show_pose = False
        self.show_crop = False


    def detect(self, frame):
        """
        Detect the hand
        :param frame: image frame
        :return: cropped image, transformation, center
        """

        hd = HandDetector(frame, self.sync['config']['fx'], self.sync['config']['fy'], importer=self.importer, refineNet=self.comrefNet)
        doHS = (self.state.value == self.STATE_INIT)
        if self.tracking.value and not numpy.allclose(self.lastcom, 0):
            loc, handsz = hd.track(self.lastcom, self.sync['config']['cube'], doHandSize=doHS)
        else:
            loc, handsz = hd.detect(size=self.sync['config']['cube'], doHandSize=doHS)

        self.lastcom = loc

        if self.state.value == self.STATE_INIT:
            self.handsizes.append(handsz)
            if self.verbose is True:
                print numpy.median(numpy.asarray(self.handsizes), axis=0)
        else:
            self.handsizes = []

        if self.state.value == self.STATE_INIT and len(self.handsizes) >= self.numinitframes:
            cfg = self.sync['config']
            cfg['cube'] = tuple(numpy.median(numpy.asarray(self.handsizes), axis=0).astype('int'))
            self.sync.update(config=cfg)
            self.state.value = self.STATE_RUN
            self.handsizes = []

        if numpy.allclose(loc, 0):
            return numpy.zeros(self.sync['config']['im_size'], dtype='float32'), numpy.eye(3), loc
        else:
            crop, M, com = hd.cropArea3D(com=loc, size=self.sync['config']['cube'],dsize=self.sync['config']['im_size'])
            com3D = self.importer.jointImgTo3D(com)
            sc = (self.sync['config']['cube'][2] / 2.)
            crop[crop == 0] = com3D[2] + sc
            crop.clip(com3D[2] - sc, com3D[2] + sc)
            crop -= com3D[2]
            crop /= sc
            return crop, M, com3D


    def addStatusBar(self, img):
        """
        Add status bar to image
        :param img: image
        :return: image with status bar
        """
        barsz = 20
        retimg = numpy.ones((img.shape[0]+barsz, img.shape[1], img.shape[2]), dtype='uint8')*255

        retimg[barsz:img.shape[0]+barsz, 0:img.shape[1], :] = img

        # FPS text
        fps = 1./(time.time()-self.lastshow)
        self.runningavg_fps.append(fps)
        avg_fps = numpy.mean(self.runningavg_fps)
        cv2.putText(retimg, "FPS {0:2.1f}".format(avg_fps), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand text
        cv2.putText(retimg, "Left" if self.hand.value == self.HAND_LEFT else "Right", (80, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand size
        ss = "HC-{0:d}".format(self.sync['config']['cube'][0])
        cv2.putText(retimg, ss, (120, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand tracking mode, tracking or detection
        cv2.putText(retimg, "T" if self.tracking.value else "D", (260, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand detection mode, COM or CNN
        if self.detection.value == self.DETECTOR_COM:
            mode = "COM"
        else:
            mode = "???"
        cv2.putText(retimg, mode, (280, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # status symbol
        if self.state.value == self.STATE_IDLE:
            col = (0, 0, 255)
        elif self.state.value == self.STATE_INIT:
            col = (0, 255, 255)
        elif self.state.value == self.STATE_RUN:
            col = (0, 255, 0)
        else:
            col = (0, 0, 255)
        cv2.circle(retimg, (5, 5), 5, col, -1)
        return retimg

    def show2(self, frame, pose,dataset):
        """
        Show depth with overlayed joints
        :param frame: depth frame
        :param pose: joint positions
        :return: image
        """

        # plot depth image with annotations
        imgcopy = frame.copy()
        # display hack to hide nd depth
        msk = np.logical_and(32001 > imgcopy, imgcopy > 0)
        msk2 = np.logical_or(imgcopy == 0, imgcopy == 32001)
        min = imgcopy[msk].min()
        max = imgcopy[msk].max()
        imgcopy = (imgcopy - min) / (max - min) * 255.
        imgcopy[msk2] = 255.
        imgcopy = imgcopy.astype('uint8')
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)
        imgcopy2 = imgcopy.copy()

        jtI = self.importer.joints3DToImg(pose)

        for i in range(jtI.shape[0]):
            cv2.circle(imgcopy, (jtI[i, 0], jtI[i, 1]), 6, (255, 255, 255), -1)

        import matplotlib
        if dataset == 'icvl':
            jointConnections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [0, 10],
                                 [10, 11], [11, 12], [0, 13], [13, 14], [14, 15]]
            jointConnectionColors = [matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 1]]]))[0, 0]]

        elif dataset =='nyu':

            jointConnections = [[0, 1], [1, 13], [2, 3], [3, 13], [4, 5], [5, 13], [6, 7], [7, 13], [8, 9], [9, 10], [10, 13],
                          [13, 11], [13, 12]]
            jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.7]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],

                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.7]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.7]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.7]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1]]]))[0, 0]]
        elif dataset == 'msra':
            jointConnections = [[0,1],[1,2],[2,3],[3,4],
          [0,5],[5,6],[6,7],[7,8],
          [0,9],[9,10],[10,11],[11,12],
          [0,13],[13,14],[14,15],[15,16],
          [0,17],[17,18],[18,19],[19,20]]
            jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.4]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.4]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.4]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.4]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.4]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0]]
        elif dataset == 'bighand':
            jointConnections = [[0, 1], [0, 2], [0, 3], [0, 4],
                                [0, 5], [1, 6], [6, 7], [7, 8],
                                [2, 9], [9, 10], [10, 11], [3, 12],
                                [12, 13], [13, 14], [4, 15], [15, 16],
                                [16, 17], [5, 18], [18, 19], [19, 20]]
            jointConnectionColors = [matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.4]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.4]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.4]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.4]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.4]]]))[0, 0],

                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.6]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.8]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 1]]]))[0, 0],

                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.6]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.8]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 1]]]))[0, 0],

                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.6]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.8]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 1]]]))[0, 0],

                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.6]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.8]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 1]]]))[0, 0],

                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.6]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.8]]]))[0, 0],
                                     matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 1]]]))[0, 0]]

        else:
            raise ValueError("Invalid number of joints")


        for i in range(len(jointConnections)):
            cv2.line(imgcopy, (jtI[jointConnections[i][0], 0], jtI[jointConnections[i][0], 1]),
                     (jtI[jointConnections[i][1], 0], jtI[jointConnections[i][1], 1]), 255.*jointConnectionColors[i], 5)

        return imgcopy

