from collections import namedtuple
from numpy.matlib import repmat
import numpy as np, matplotlib.pyplot as plt, matplotlib, cv2
import cv2
#import globalConfig

CameraOption = namedtuple('CameraOption', ['focal_x', 'focal_y', 'center_x', 'center_y', 'width', 'height', 'far_point'])
# Frame = namedtuple('Frame', ['dm', 'skel', 'crop_dm', 'crop_skel', 'file_name'])
# Frame.__new__.__defaults__ = (None,)*len(Frame._fields)

figColor = [(19,69,139),
            (51,51,255),
            (51,151,255),
            (51,255,151),
            (255,255,51),
            (255,51,153),
            (0,255,0)
           ]
nyuColorIdx = [1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6 + [0]*6
nyuFigColorIdx = [1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6
# for l in globalConfig.nyuKeepList: 
    # nyuColorIdx[l] = 6
icvlColorIdx = [0] + [1]*3 + [2]*3 + [3]*3 + [4]*3 + [5]*3
msraColorIdx = [0] + [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4

initFigBone = lambda startIdx,jntNum,color : \
        [(sIdx,eIdx,color) for sIdx, eIdx in\
         zip(range(startIdx,startIdx+jntNum-1),range(startIdx+1,startIdx+jntNum))]
def flattenBones(bones):
    b = []
    for bb in bones:
        b += bb
    return b
nyuBones = flattenBones([initFigBone(b*6,6,figColor[b+1]) for b in range(5)])
nyuBones14 = flattenBones([initFigBone(b*2,2,(255,51,153)) for b in range(5)]) + [(-4,-5,(255,51,153))] +[(b*2+1,-1,(255,51,153)) for b in range(4)]
icvlBones = flattenBones([initFigBone(b*3+1,3,figColor[b+1]) for b in range(5)])
msraBones = flattenBones([initFigBone(b*4+1,4,figColor[b+1]) for b in range(5)])


class Camera(object):
    intel = [241.42, 241.42, 160, 120, 320, 240, 32001]
    kinect = [588.235, 587.084, 320, 240, 640, 480, 2001]
    
    dataset = 'NYU'
    #set as default
    if dataset == 'NYU':
        current = CameraOption(*kinect)
    elif dataset == 'ICVL':
        current = CameraOption(*intel)
    elif dataset == 'MSRA':
        current = CameraOption(*intel)
    else:
        print globalConfig.dataset
        raise NotImplementedError('Unknown dataset %s'%globalConfig.dataset)

    focal_x = current.focal_x
    focal_y = current.focal_y
    center_x = current.center_x
    center_y = current.center_y
    width = current.width
    height = current.height
    far_point = current.far_point

    @classmethod
    def setCamera(cls_obj, camera_type):
        if camera_type.upper() == 'INTEL':
            cls_obj.current = CameraOption(*cls_obj.intel)
        elif camera_type.upper() == 'KINECT':
            cls_obj.current = CameraOption(*cls_obj.kinect)
        else:
            raise ValueError('the input type is incorrect')
        cls_obj.focal_x = cls_obj.current.focal_x
        cls_obj.focal_y = cls_obj.current.focal_y
        cls_obj.center_x = cls_obj.current.center_x
        cls_obj.center_y = cls_obj.current.center_y
        cls_obj.width = cls_obj.current.width
        cls_obj.height = cls_obj.current.height
        cls_obj.far_point = cls_obj.current.far_point
        print 'current camera type is set as {} with {}'.format(camera_type.upper(), cls_obj.current)

    @classmethod
    def to3D(cls_obj, pt2):
        pt3 = np.zeros((3), np.float32)
        pt3[0] = (pt2[0] - cls_obj.center_x)*pt2[2] / cls_obj.focal_x
        pt3[1] = (cls_obj.center_y - pt2[1])*pt2[2] / cls_obj.focal_y
        pt3[2] = pt2[2]
        return pt3

    @classmethod
    def to2D(cls_obj, pt3):
        pt2 = np.zeros((3), np.float32)
        pt2[0] =  pt3[0]*cls_obj.focal_x / pt3[2] + cls_obj.center_x
        pt2[1] = -pt3[1]*cls_obj.focal_y / pt3[2] + cls_obj.center_y
        pt2[2] = pt3[2]
        return pt2

class Frame(object):
    skel_norm_ratio = 50.0
    def __init__(self, dm = None, dms = None, skel = None, com2D = None, flag = None):
        if not isinstance(com2D, np.ndarray):
            (self.crop_dm, self.trans, self.com3D) = dm.Detector()
        else:
            (self.crop_dm, self.trans, self.com3D)=  dm.cropArea3D(dm.dmData, com2D)
            (self.crop_dms, self.transs, self.com3Ds)=  dms.cropArea3D(dms.dmData, com2D)
        self.normDm()
	self.dm = dm.dmData
	try:
          self.normDms()
	  self.dms = dms.dmData
	except:
	  pass

        if isinstance(skel, np.ndarray):
            if len(skel)%3 != 0:
                raise ValueError('invalid length of the skeleton mat')
            jntNum = len(skel)/3
            self.skel = skel.astype(np.float32)
            #crop_skel is the training label for neurual network, normalize wrt com3D
            self.crop_skel = (self.skel - repmat(self.com3D, 1, jntNum))[0]
            self.crop_skel = self.crop_skel.astype(np.float32)
            self.normSkel()
	    self.skel2 = np.flipud(self.crop2D()) #np.flipud
	    #print(self.skel2.shape)
	#self.showAnnotatedSample()
	#self.saveAnnotatedSample('salam.jpg')

    # save only the norm_dm and norm_skel for training, clear all initial size data
    def saveOnlyForTrain(self):
        self.dm = None
	self.dms = None
        self.crop_dm = None
        self.crop_dms = None
        self.skel = None
	self.skel2 = None
        self.crop_skel = None
        # self.trans = None
        # self.com3D = None

    def normDm(self):
        self.norm_dm = self.crop_dm.copy()
        m = self.norm_dm.max()
        if m == 0:
            return
        self.norm_dm /= m
        self.norm_dm -= 0.5

    def normDms(self):
        self.norm_dms = self.crop_dms.copy()
        m = self.norm_dms.max()
        if m == 0:
            return
        self.norm_dms /= m
        self.norm_dms -= 0.5


    def normSkel(self):
        self.norm_skel = self.crop_skel.copy() / self.skel_norm_ratio

    # inverse process, input: norm_skel, output: crop_skel, skel
    def setNormSkel(self, norm_skel):
        if len(norm_skel)%3 != 0:
            raise ValueError('invalid length of the skeleton mat')
        jntNum = len(norm_skel)/3
        self.norm_skel = norm_skel.copy().astype(np.float32)
        self.crop_skel = norm_skel.copy()*self.skel_norm_ratio
        self.com3D = np.zeros([3])
        self.com3D[2] = 200
        self.skel = (self.crop_skel + repmat(self.com3D, 1, jntNum))[0]
        self.skel = self.skel.astype(np.float32)

    def setCropSkel(self, crop_skel):
        if len(crop_skel)%3 != 0:
            raise ValueError('invalid length of the skeleton mat')
        jntNum = len(crop_skel)/3
        self.crop_skel = crop_skel.astype(np.float32)
        self.skel = (self.crop_skel + repmat(self.com3D, 1, jntNum))[0]
        self.skel = self.skel.astype(np.float32)
        self.normSkel()


    def setSkel(self, skel):
        if len(skel)%3 != 0:
            raise ValueError('invalid length of the skeleton mat')
        jntNum = len(skel)/3
        self.skel = skel.astype(np.float32)
        #crop_skel is the training label for neurual network, normalize wrt com3D
        self.crop_skel = (self.skel - repmat(self.com3D, 1, jntNum))[0]
        self.crop_skel = self.crop_skel.astype(np.float32)
        self.normSkel()

    def saveAnnotatedSample(self, path):
        skel2 = self.crop2D()
        skel2 = skel2.reshape(-1, 3)
        for i, pt in enumerate(skel2):
            skel2[i] = Camera.to2D(pt)
        print 'current camera option={}'.format(Camera.focal_x)

        skel = self.skel
        skel.shape = (-1,3)

        dm = self.norm_dm.copy()
        dm[dm == Camera.far_point] = 0
        fig = plt.figure()
        ax = fig.add_subplot(121)
        img = dm.copy() 
        img = img - img.min()
        img *= 255/img.max()
	print(skel2[0])
        for pt in skel2:
            cv2.circle(img, (pt[0], pt[1]), 2, (255,0,0), -1)
        cv2.imwrite(path, img)
            
    def showAnnotatedSample(self):
        fig = plt.figure()
        fig.suptitle('annoated example')
        if self.dm is not None and self.skel is not None:
            skel2 = self.skel.copy()
            skel2 = skel2.reshape(-1, 3)
            for i, pt in enumerate(skel2):
                skel2[i] = Camera.to2D(pt)

            print 'current camera option={}'.format(Camera.focal_x)

            skel = self.skel
            skel.shape = (-1,3)

            dm = self.dm.copy()
            dm[dm == Camera.far_point] = 0
            ax = fig.add_subplot(121)
            ax.imshow(dm, cmap=matplotlib.cm.jet)
            ax.scatter(skel2[:5,0], skel2[:5,1], c='r')
            ax.set_title('initial')

        if self.crop_dm is not None and self.crop_skel is not None:
            skel2 = self.crop2D()

            ax = fig.add_subplot(122)
            ax.imshow(self.crop_dm, cmap=matplotlib.cm.jet)
            ax.scatter(skel2[:5,0], skel2[:5,1], c='r')
            ax.set_title('cropped')

        if self.norm_dm is not None and self.norm_skel is not None:
            skel2 = self.crop2D()
            ax = fig.add_subplot(122)
            ax.imshow(self.norm_dm, cmap=matplotlib.cm.jet)
            ax.scatter(skel2[:5,0], skel2[:5,1], c='r')
            ax.set_title('normed')
        plt.show()

    def crop2D(self):
        self.crop_skel = self.norm_skel * np.float32(50.0)
        skel = self.crop_skel.copy()
        jntNum = len(skel)/3
        skel = skel.reshape(-1, 3)
        skel += repmat(self.com3D, jntNum, 1)
        for i, jnt in enumerate(skel):
            jnt = Camera.to2D(jnt)
            pt = np.array([jnt[0],jnt[1], 1.0], np.float32).reshape(3,1)
            pt = self.trans*pt
            skel[i,0], skel[i,1] = pt[0], pt[1]
        return skel

    def full2D(self):
        '''
        2D transformation of the cropped_pt(the estimated one) to the initial
        size image
        '''
        skel = self.skel.copy()
        skel = skel.reshape(-1,3)

        for i, jnt in enumerate(skel):
            jnt = Camera.to2D(jnt)
            skel[i,0], skel[i,1] = jnt[0], jnt[1]
        return skel

    def visualizeCrop(self, norm_skel = None):
        img = self.norm_dm.copy()
        img = (img+0.5)*255.0
        colorImg = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if norm_skel is None:
            return colorImg

        self.setNormSkel(norm_skel)
        skel2D = self.crop2D()
        for pt in skel2D:
            cv2.circle(colorImg, (pt[0], pt[1]), 2, (0,0,255), -1)
        return colorImg


    def visualizeFull(self, norm_skel = None):
        img = self.dm.copy()
        img[img >= Camera.far_point] = 0
        img = img*(256/img.max())
        colorImg = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if norm_skel is None:
            return colorImg

        self.setNormSkel(norm_skel)
        skel2D = self.full2D()
        for pt in skel2D:
            cv2.circle(colorImg, (pt[0], pt[1]), 5, (0,0,255), -1)
        return colorImg

def vis_pose(normed_vec):
    import depth 
    origin_pt = np.array([0,0,depth.DepthMap.invariant_depth])
    vec = normed_vec.copy()*50.0
    vec.shape = (-1,3)

    offset_x = Camera.center_x - depth.DepthMap.size2[0]/2
    offset_y = Camera.center_y - depth.DepthMap.size2[1]/2
    
    img = np.ones((depth.DepthMap.size2[0], depth.DepthMap.size2[1]))*255
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
    for idx, pt3 in enumerate(vec):
        pt = Camera.to2D(pt3+origin_pt)
        pt = (pt[0]-offset_x, pt[1]-offset_y)
        cv2.circle(img, (int(pt[0]), int(pt[1])),2, (255,0,0), -1)
    return img

def vis_normed_pose(normed_vec, img=None):
    import depth
    pt2 = projectNormPose3D(normed_vec)

    if not type(img) is np.ndarray:
        img = np.ones((depth.DepthMap.size2[0], depth.DepthMap.size2[1]))*255

    img = img.reshape(depth.DepthMap.size2[0], depth.DepthMap.size2[1])
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
    for idx, pt in enumerate(pt2):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1)
    return img
