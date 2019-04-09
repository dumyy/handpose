import tensorflow as tf


def unnorm(hand_outs,subComs,subCubes,joint):
    def fn(elems):
        hand_out,subCom,subCube=elems[0],elems[1],elems[2]
        hand_=tf.multiply(hand_out,subCube[2]/2.)+tf.tile(tf.expand_dims(subCom,0),(joint,1))
        return [hand_,subCom,subCube]
    hands_xyz,_,_=tf.map_fn(fn,[hand_outs,subComs,subCubes])
    return hands_xyz

def back2Dori(hands_xyz,cfg):
    def fn(elems):
        hand_xyz,cfg=elems[0],elems[1]
        hand_xyz=hand_xyz/tf.tile(tf.expand_dims(hand_xyz[:,2],1),(1,3))
        uvd0=tf.matmul(cfg,tf.transpose(hand_xyz,(1,0)))
        #uvd0=tf.transpose(uvd0,(1,0))
        return [uvd0,cfg]
    uvds,_=tf.map_fn(fn,[hands_xyz,cfg])
    return uvds

def back2Dnew(uvds,subMs):
    def fn(elems):
        uvd,subM=elems[0],elems[1]
        uvd0=tf.matmul(subM,uvd)
        uvd0=tf.transpose(uvd0,(1,0))
        return [uvd0,subM]
    uvds, _=tf.map_fn(fn,[uvds,subMs])
    return uvds

def genHtmap(uvds,kernels):
    def fn(elems):
        uvd,kernel=elems[0],elems[1]
        uvd_pts = tf.reshape(uvd, (-1, 3))
        num_pt = uvd_pts.shape[0]
        num_pt_op = tf.to_int64(num_pt)

        nn = tf.range(num_pt, dtype=tf.int64)
        nn = tf.reshape(nn, (-1, 1))

        xx = uvd_pts[:, 0]
        xx = tf.clip_by_value(xx, 0, 24 - 1)
        xx = tf.to_int64(xx)
        xx = tf.reshape(xx, (-1, 1))

        yy = uvd_pts[:, 1]
        yy = tf.clip_by_value(yy, 0, 24 - 1)
        yy = tf.to_int64(yy)
        yy = tf.reshape(yy, (-1, 1))
        indices = tf.concat([nn, yy, xx], axis=1)

        val = 1.0
        raw_hm = tf.sparse_to_dense(sparse_indices=indices,
                                    output_shape=[num_pt_op, 24, 24],
                                    sparse_values=val)
        raw_hm = tf.expand_dims(raw_hm, axis=[-1])
        raw_hm = tf.cast(raw_hm, tf.float32)
        raw_hm=tf.nn.conv2d(raw_hm,tf.reshape(kernel,(3,3,1,1)),strides=(1,1,1,1),padding='SAME',data_format='NHWC')
        raw_hm=tf.squeeze(raw_hm)
        return [raw_hm,kernel]
    rets,_=tf.map_fn(fn,[uvds,kernels])
    rets=tf.transpose(rets,(0,2,3,1))
    return rets

