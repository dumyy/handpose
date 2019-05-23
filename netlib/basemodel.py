import tensorflow as tf
import tensorflow.contrib.slim as slim
from netutil import resnet_v1
from netutil import resnet_utils
bottleneck=resnet_v1.bottleneck

def basenet2(inp,kp=0.5,is_training=True,outdims=(14,9,5)):
    '''

    :param inp: input data
    :param kp: droupout keep rate
    :param is_training: is_trainging?
    :param outdims: (hand_num,palm_num,finger_num)
    :return: output
    '''
    with tf.name_scope("bone_net"):
        blocks = [
            resnet_v1.resnet_v1_block('block1', base_depth=16, num_units=3, stride=2),
            resnet_v1.resnet_v1_block('block2', base_depth=32, num_units=4, stride=2),
            resnet_v1.resnet_v1_block('block3', base_depth=64, num_units=6, stride=2),
            resnet_v1.resnet_v1_block('block4', base_depth=64, num_units=3, stride=2),
        ]
        net = resnet_utils.conv2d_same(
            inp, 32, 5, stride=2, scope='conv1')
        # net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]],)
        net = slim.max_pool2d(
            net, [3, 3], stride=1, padding='SAME', scope='pool1')
        with slim.arg_scope([resnet_v1.resnet_v1],
                            is_training=None, global_pool=False, include_root_block=False):
            net1, _ = resnet_v1.resnet_v1(net, blocks[0:1], scope='nn1', )
            net2, _ = resnet_v1.resnet_v1(net1, blocks[1:2], scope='nn2', )
            net3, _ = resnet_v1.resnet_v1(net2, blocks[2:3], scope='nn3', )
            net4, _ = resnet_v1.resnet_v1(net3, blocks[3:4], scope='nn4', )
        feature_maps = [net1, net2, net3,net4]

    with tf.name_scope("elem_net"):
        global_fms = []
        last_fm = None
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope("global_net"):
            for i, block in enumerate(reversed(feature_maps)):
                lateral = slim.conv2d(block, 256, [1, 1],
                                      weights_initializer=initializer,
                                      padding='SAME', activation_fn=tf.nn.relu,
                                      scope='lateral/res{}'.format(5 - i))
                if last_fm is not None:
                    sz = tf.shape(lateral)
                    upsample = tf.image.resize_bilinear(last_fm, (sz[1], sz[2]),
                                                        name='upsample/res{}'.format(5 - i))
                    upsample = slim.conv2d(upsample, 256, [1, 1],
                                           weights_initializer=initializer,
                                           padding='SAME', activation_fn=None,
                                           scope='merge/res{}'.format(5 - i))
                    last_fm = upsample + lateral
                else:
                    last_fm = lateral
                global_fms.append(last_fm)
        global_fms.reverse()

    with tf.name_scope("cacsed"):

        hand3_map_ = global_fms[-3]

        palm_map = bottleneck(hand3_map_, 256, 128, stride=1, scope="palm_bottleneck")

        ht_palm = slim.conv2d(palm_map, 256, 1, 1, activation_fn=tf.nn.relu)
        ht_palm_out_ = slim.conv2d(ht_palm, num_outputs=outdims[1], kernel_size=(3, 3), activation_fn=None)
        ht_palm_out = tf.image.resize_bilinear(ht_palm_out_, (24, 24))

        fing_map = bottleneck(hand3_map_, 256, 128, stride=1, scope="fing_bottleneck")

        ht_fing = slim.conv2d(fing_map, 256, 1, 1, activation_fn=tf.nn.relu)
        ht_fing_out_ = slim.conv2d(ht_fing, num_outputs=outdims[2], kernel_size=(3, 3), activation_fn=None)
        ht_fing_out = tf.image.resize_bilinear(ht_fing_out_, (24, 24))

        res_fing_map = hand3_map_ - palm_map

        end_fing_map = tf.concat([fing_map , res_fing_map],3)

        end_fing_map_ = bottleneck(end_fing_map, 256, 128, stride=1, scope="end_fing_bottleneck")
        end_fing_map_pooling = slim.max_pool2d(end_fing_map_, 2)

        end_fing_ = slim.flatten(end_fing_map_pooling)
        end_fing_ = slim.fully_connected(end_fing_, 1024, activation_fn=tf.nn.relu)
        end_fing_ = slim.dropout(end_fing_, keep_prob=kp, is_training=is_training)
        end_fing_ = slim.fully_connected(end_fing_, 1024, activation_fn=tf.nn.relu)
        end_fing_ = slim.dropout(end_fing_, keep_prob=kp, is_training=is_training)
        end_fing_out = slim.fully_connected(end_fing_, outdims[2]*3, activation_fn=None)

        res_palm_map = hand3_map_ - fing_map


        end_palm_map = tf.concat([palm_map , res_palm_map],3)

        end_palm_map_ = bottleneck(end_palm_map, 256, 128, stride=1, scope="end_palm_bottleneck")
        end_palm_map_pooling = slim.max_pool2d(end_palm_map_, 2)

        end_palm_ = slim.flatten(end_palm_map_pooling)
        end_palm_ = slim.fully_connected(end_palm_, 1024, activation_fn=tf.nn.relu)
        end_palm_ = slim.dropout(end_palm_, keep_prob=kp, is_training=is_training)
        end_palm_ = slim.fully_connected(end_palm_, 1024, activation_fn=tf.nn.relu)
        end_palm_ = slim.dropout(end_palm_, keep_prob=kp, is_training=is_training)
        end_palm_out = slim.fully_connected(end_palm_, outdims[1]*3, activation_fn=None)

        end_hand = tf.concat([end_palm_, end_fing_], 1)
        # end_hand_ = slim.fully_connected(end_hand, 1024, activation_fn=tf.nn.relu)
        # end_hand_ = slim.dropout(end_hand_, keep_prob=kp, is_training=is_training)
        end_hand_out = slim.fully_connected(end_hand, outdims[0]*3, activation_fn=None)

        comb_ht_out = [ht_palm_out, ht_fing_out]
        comb_hand_out = [end_palm_out, end_fing_out]
        hand_out = end_hand_out


    with tf.name_scope("htmap"):
        ht_map=global_fms[-4]
        ht_map=bottleneck(ht_map,256,128,stride=1,scope="htmap_bottleneck")
        ht_out=slim.conv2d(ht_map,num_outputs=outdims[0],kernel_size=(3,3),stride=1,activation_fn=None)

    return comb_ht_out,comb_hand_out,hand_out,ht_out
