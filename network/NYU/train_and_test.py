import sys
sys.path.append('../../')#add root directory
from data.importers import NYUImporter
from util.handdetector import HandDetector
from data.transformations import transformPoints2D

import numpy as np
from util.preprocess import augmentCrop,norm_dm,joints_heatmap_gen
import tensorflow as tf
rng=np.random.RandomState(23455)

from netlib.basemodel import basenet2

train_root='/home/dumyy/data/nyu/dataset/'
di_train = NYUImporter(train_root, cacheDir='../../cache/NYU/',refineNet=None,allJoints=False)

Seq_train = di_train.loadSequence('train', rng=rng, shuffle=True, docom=False,cube=None)
train_num=len(Seq_train.data)
cubes_train = np.asarray([d.cube for d in Seq_train.data], 'float32')
coms_train = np.asarray([d.com for d in Seq_train.data], 'float32')
Ms_train = np.asarray([d.T for d in Seq_train.data], dtype='float32')
gt3Dcrops_train = np.asarray([d.gt3Dcrop for d in Seq_train.data], dtype='float32')
imgs_train = np.asarray([d.dpt.copy() for d in Seq_train.data], 'float32')


di_test = NYUImporter(train_root, cacheDir='../../cache/NYU/',refineNet=None,allJoints=False)
Seq_test = di_test.loadSequence('test', rng=rng, shuffle=False, docom=False,cube=(250,250,250))
test_num=len(Seq_test.data)
cubes_test = np.asarray([d.cube for d in Seq_test.data], 'float32')
coms_test = np.asarray([d.com for d in Seq_test.data], 'float32')
gt3Dcrops_test = np.asarray([d.gt3Dcrop for d in Seq_test.data], dtype='float32')
imgs_test = np.asarray([d.dpt.copy() for d in Seq_test.data], 'float32')
test_data=np.ones_like(imgs_test)
test_label=np.ones_like(gt3Dcrops_test)
for it in range(test_num):
    test_data[it]=norm_dm(imgs_test[it],coms_test[it],cubes_test[it])
    test_label[it]=gt3Dcrops_test[it]/(cubes_test[it][0]/2.)
test_data=np.expand_dims(test_data,3)
test_label=np.reshape(test_label,(-1,42))

print "training data {}".format(imgs_train.shape[0])
print "testing data {}".format(imgs_test.shape[0])
print "done"



visual_aug=False

hand_edges=[[0,1],[2,3],[4,5],[6,7],[8,9],[9,10],[13,10],[13,1],[13,3],[13,5],[13,7],[13,11],[13,12]]


hd=HandDetector(imgs_train[0].copy(),abs(588.03),abs(587.07),di_train,refineNet=None)

inputs=tf.placeholder(dtype=tf.float32,shape=(None,96,96,1))
label=tf.placeholder(dtype=tf.float32,shape=(None,42))
gt_ht=tf.placeholder(dtype=tf.float32,shape=(None,24,24,14))
is_train=tf.placeholder(dtype=tf.bool,shape=None)

kp=tf.placeholder(dtype=tf.float32,shape=None)

batch_size=128
last_e=100
outdims=(14,9,5)
#################################################################
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

fn=layers.l2_regularizer(1e-5)
fn0=tf.no_regularizer

with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_regularizer=fn,
                    biases_regularizer=fn0,
                    normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm],
                        is_training=is_train,
                        updates_collections=None,
                        decay=0.9,
                        center=True,
                        scale=True,
                        epsilon=1e-5):
        pred_comb_ht, pred_comb_hand, pred_hand, pred_ht=basenet2(inputs,kp=kp,is_training=is_train,outdims=outdims)


gt_fing_ht=gt_ht[:,:,:,0:10][:,:,:,0::2]
gt_palm_ht = tf.concat((gt_ht[:,:,:,0:10][:, :,:,1::2], gt_ht[:,:,:, 10:]), 3)

label1=tf.reshape(label,(-1,14,3))
gt_fing = tf.reshape(label1[:, 0:10, :][:, 0::2, :], (-1, 5 * 3))
gt_palm = tf.reshape(tf.concat((label1[:, 0:10, :][:, 1::2, :], label1[:, 10:, :]), 1),(-1,9*3))


loss_ht=tf.nn.l2_loss((pred_ht-gt_ht))/batch_size
loss_hand=tf.nn.l2_loss((pred_hand-label))/batch_size

loss_palm_ht=tf.nn.l2_loss((pred_comb_ht[0]-gt_palm_ht))/batch_size
loss_fing_ht=tf.nn.l2_loss((pred_comb_ht[1]-gt_fing_ht))/batch_size

loss_palm=tf.nn.l2_loss((pred_comb_hand[0]-gt_palm))/batch_size
loss_fing=tf.nn.l2_loss((pred_comb_hand[1]-gt_fing))/batch_size

weight_decay = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
weight_decay = tf.add_n(weight_decay)


loss=0.5*(0.01*(loss_ht+loss_palm_ht+loss_fing_ht)+loss_hand+loss_palm+loss_fing)+weight_decay

#################################################################
def getMeanError(gt, joints):
    return np.nanmean(np.nanmean(np.sqrt(np.square(gt - joints).sum(axis=2)), axis=1))
#################################################################


lr = tf.Variable((1e-3), dtype=tf.float32, trainable=False)
global_step = tf.Variable(0, trainable=False)


optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step)

tf.summary.scalar("loss",loss)
tf.summary.scalar('lr',lr)

summ=tf.summary.merge_all()
saver=tf.train.Saver(max_to_keep=15)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #writer = tf.summary.FileWriter('../../tensorboard.events9/', sess.graph)

    training_batch = (zip(range(0, train_num, batch_size),
                          range(batch_size, train_num + 1, batch_size)))
    for itrain in range(200):
        kpv = 0.6
        c = list(zip(cubes_train, coms_train, Ms_train, gt3Dcrops_train, imgs_train))
        rng.shuffle(c)
        cubes_train, coms_train, Ms_train, gt3Dcrops_train, imgs_train = zip(*c)
        cubes_train = np.asarray(cubes_train)
        coms_train = np.asarray(coms_train)
        Ms_train = np.asarray(Ms_train)
        gt3Dcrops_train = np.asarray(gt3Dcrops_train)
        imgs_train = np.asarray(imgs_train)

        lr_update = tf.assign(lr, (1e-3) * 0.96 ** itrain)
        sess.run(lr_update)

        for start, end in training_batch:
            subdata = imgs_train[start:end].copy()
            subcom = coms_train[start:end].copy()
            subcube = cubes_train[start:end].copy()
            subM = Ms_train[start:end].copy()
            subgt3Dcrop = gt3Dcrops_train[start:end].copy()
            resdata=np.ones_like(subdata)
            resgt3D=np.ones_like(subgt3Dcrop)
            hts=np.zeros(shape=(batch_size,24,24,14))
            jtIs=np.zeros(shape=(batch_size,42))

            for idx in range(batch_size):
                dm = norm_dm(subdata[idx], subcom[idx], subcube[idx])
                s = augmentCrop(dm, subgt3Dcrop[idx], di_train.joint3DToImg(subcom[idx]),
                                subcube[idx], subM[idx],['com', 'rot', 'none','sc'], hd, False, rng=rng)
                resdata[idx] = s[0]
                resgt3D[idx] = s[2]
                mode=s[7]
                gt3D_ = resgt3D[idx] * subcube[idx][0] / 2. + subcom[idx]
                jtI_ = transformPoints2D(di_train.joints3DToImg(gt3D_), subM[idx])
                jtI_=np.reshape(jtI_,(1,42))
                ht_=joints_heatmap_gen([1],jtI_,(24,24),points=14)
                hts[idx]=np.transpose(ht_,(0,2,3,1))/255.


            feed_dict = {inputs: resdata.reshape(-1, 96, 96, 1),
                         label: resgt3D.reshape(-1, 42),
                         gt_ht:hts,
                         is_train:True,
                         kp:kpv
                         }
            _, losses,summs,steps = sess.run([optimizer, loss,summ,global_step], feed_dict=feed_dict)

            #writer.add_summary(summs,steps)
            print itrain,start/batch_size,losses,mode

        pred_norm=[]
        loopv = test_num // batch_size
        other = test_data[loopv * batch_size:]
        for itest in xrange(loopv + 1):
            if itest < loopv:
                start = itest * batch_size
                end = (itest + 1) * batch_size
                feed_dict = {inputs: test_data[start:end],is_train:False,kp:kpv}
            else:
                feed_dict = {inputs: other,is_train:False,kp:kpv}
            [pred_] = sess.run([pred_hand], feed_dict=feed_dict)
            pred_norm.append(pred_)
        norm_hands = np.concatenate(pred_norm, 0).reshape(-1, 14, 3)
        pred_hands = norm_hands * np.tile(np.expand_dims(cubes_test / 2., 1), (1, 14, 1)) + \
                     np.tile(np.expand_dims(coms_test, 1),(1, 14, 1))
        gt_hands = test_label.reshape(-1, 14, 3) * np.tile(np.expand_dims(cubes_test / 2., 1), (1, 14, 1)) + \
                     np.tile(np.expand_dims(coms_test, 1), (1, 14, 1))
        meane = getMeanError(gt_hands, pred_hands)

        logt = open('../../log/mean.error.epoch/log_nyu.txt', 'a+')
        logt.write('epoch {}, mean error {}'.format(itrain,meane))
        logt.write('\n')
        logt.close()

        if last_e>=meane:
            last_e=meane
            saver.save(sess,'../../model/crossInfoNet_nyu.ckpt')
            logt = open('../../log/mean.error.epoch/log_nyu.txt', 'a+')
            logt.write("*********************")
            logt.write('\n')
            logt.write('current best epoch is {}, mean error is {}'.format(itrain,last_e))
            logt.write('\n')
            logt.write("*********************")
            logt.write('\n')
            logt.close()







