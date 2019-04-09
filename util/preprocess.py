import cv2
import numpy as np

def norm_dm(img, com, cube):
    img_=img.copy()
    img_[img_ == 0] = com[2] + (cube[2] / 2.)
    img_ -= com[2]
    img_ /= (cube[2] / 2.)
    return img_
def norm_dm1(dm,com,cube):
    max_depth = com[2] + cube[2] * 0.5
    min_depth = com[2] - cube[2] * 0.5
    mask = np.logical_and(np.less(dm, max_depth), np.greater(dm, min_depth - cube[2] * 0.5))
    normed_dm = np.where(mask, 1-np.divide(dm - min_depth, cube[2]), -1.0 * np.ones_like(dm))
    return normed_dm

def augmentCrop(img, gt3Dcrop, com, cube, M, aug_modes, hd, normZeroOne=False, sigma_com=None,
                sigma_sc=None, rot_range=None, rng=None):

    """
    Commonly used function to augment hand poses
    :param img: image
    :param gt3Dcrop: 3D annotations
    :param com: center of mass in image coordinates (x,y,z)
    :param cube: cube
    :param aug_modes: augmentation modes
    :param hd: hand detector
    :param normZeroOne: normalization
    :param sigma_com: sigma of com noise
    :param sigma_sc: sigma of scale noise
    :param rot_range: rotation range in degrees
    :return: image, 3D annotations, com, cube
    """
    #print(img.shape)
    assert len(img.shape) == 2
    assert isinstance(aug_modes, list)

    if sigma_com is None:
        sigma_com = 10.

    if sigma_sc is None:
        sigma_sc = 0.05

    if rot_range is None:
        rot_range = 180.

    if normZeroOne is True:
        img = img * cube[2] + (com[2] - (cube[2] / 2.))
    else:
        img = img * (cube[2] / 2.) + com[2]
    premax = img.max()
    mode = rng.randint(0, len(aug_modes))
    off = rng.randn(3) * sigma_com  # +-px/mm
    rot = rng.uniform(-rot_range, rot_range)
    sc = abs(1. + rng.randn() * sigma_sc)


    if aug_modes[mode] == 'com':
    #print('aug com', off)
        rot = 0.
        sc = 1.
        imgD, new_joints3D, com, M = hd.moveCoM(img.astype('float32'), cube,
                                                com, off, gt3Dcrop, M, pad_value=0)
        curLabel = new_joints3D / (cube[2] / 2.)
    elif aug_modes[mode] == 'rot':
    #print('aug rot', rot)
        off = np.zeros((3,))
        sc = 1.
        imgD, new_joints3D, rot = hd.rotateHand(img.astype('float32'), cube,
                                                com, rot, gt3Dcrop, pad_value=0)
        curLabel = new_joints3D / (cube[2] / 2.)
    elif aug_modes[mode] == 'sc':
        off = np.zeros((3,))
        rot = 0.
        imgD, new_joints3D, cube, M = hd.scaleHand(img.astype('float32'), cube,
                                                   com, sc, gt3Dcrop, M, pad_value=0)
        curLabel = new_joints3D / (cube[2] / 2.)
    elif aug_modes[mode] == 'none':
        off = np.zeros((3,))
        sc = 1.
        rot = 0.
        imgD = img
        curLabel = gt3Dcrop / (cube[2] / 2.)
    else:
        raise NotImplementedError()

    if normZeroOne is True:
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= (com[2] - (cube[2] / 2.))
        imgD /= cube[2]
    else:
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)


    return imgD, None, curLabel, np.asarray(cube), com, np.array(M, dtype='float32'), rot, aug_modes[mode]



def joints_heatmap_gen(data, label, tar_size=(96,96), ori_size=(96,96), points=16,
                       return_valid=False, gaussian_kernel=(3,3)):
    if return_valid:
        valid = np.ones((len(data), points), dtype=np.float32)
    ret = np.zeros((len(data), points, tar_size[0], tar_size[1]), dtype='float32')
    for i in range(len(ret)):
        for j in range(points):
            if label[i][j*3] < 0 or label[i][j*3+1] < 0:
                continue
            label[i][j*3+1] = min(label[i][j*3+1], ori_size[0] - 1)
            label[i][j*3] = min(label[i][j*3], ori_size[1] - 1)
            ret[i][j][int(label[i][j*3+1] * tar_size[0] / ori_size[0])][
                int(label[i][j*3] * tar_size[1] / ori_size[1])] = 1
    for i in range(len(ret)):
        for j in range(points):
            ret[i, j] = cv2.GaussianBlur(ret[i, j], gaussian_kernel, 0)
            # plt.imshow(ret[i,j])
            # plt.show()
    for i in range(len(ret)):
        for j in range(points):
            am = np.amax(ret[i][j])
            if am <= 1e-8:
                if return_valid:
                    valid[i][j] = 0.
                continue
            ret[i][j] /= am /255
    if return_valid:
        return ret, valid
    else:
        return ret

