from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer import gcn
import matplotlib.pyplot as plt


@tf.function
def build_adjacency_hamming(tensor_in):
    """
    Hamming-distance-based graph. It is self-connected.
    :param tensor_in: [N D]
    :return:
    """
    code_length = tf.cast(tf.shape(tensor_in)[1], tf.float32)
    m1 = tensor_in - 1
    c1 = tf.matmul(tensor_in, m1, transpose_b=True)
    c2 = tf.matmul(m1, tensor_in, transpose_b=True)
    normalized_dist = tf.math.abs(c1 + c2) / code_length
    return tf.pow(1 - normalized_dist, 1.4)

@tf.function
def hamming_split(t1):
      code_length = tf.cast(tf.shape(t1)[1], tf.float32)
      m1 = t1 - 1
      c1 = tf.matmul(t1, m1, transpose_b=True)
      c2 = tf.matmul(m1, t1, transpose_b=True)
      t1 = 1 - tf.math.abs(c1 + c2) / code_length 
      return t1

@tf.function
def build_adjacency_hamming_adapt(tensor_in):
    """
    Hamming-distance-based graph. It is self-connected.
    :param tensor_in: [N D]
    :return:
    """
    
    cl = tf.shape(tensor_in)[1]
    idxs = tf.range(cl)
    ridxs = tf.random.shuffle(idxs)
    rinput = tf.gather(tensor_in,ridxs,axis=1)

    #rinput = hamming_split(rinput)

    #t = tf.stack(tf.split(tensor_in,4,1))
    
    #normalized_dist = tf.map_fn(fn=hamming_split,elems=t)
    #y=(x), y = x**2, y=1- reshape(1-x)**2
    #normalized_dist = normalized_dist/tf.math.reduce_max(normalized_dist)
    #normalized_dist = 1-tf.pow(normalized_dist+.0001, .5)
    normalized_dist = tf.pow(rinput, 1.4)

    #means = tf.math.reduce_mean(normalized_dist,axis=0)
    #var = tf.math.reduce_variance(normalized_dist,axis=0)


    return rinput#tf.stack([means,var])


# noinspection PyAbstractClass
class TwinBottleneck(tf.keras.layers.Layer):
    def __init__(self, bbn_dim, cbn_dim, **kwargs):
        super().__init__(**kwargs)
        self.bbn_dim = bbn_dim
        self.cbn_dim = cbn_dim
        self.gcn = gcn.GCNLayer(cbn_dim)

    # noinspection PyMethodOverriding
    def call(self, bbn, cbn):
        adj = build_adjacency_hamming(bbn)
        return tf.nn.sigmoid(self.gcn(cbn, adj))
