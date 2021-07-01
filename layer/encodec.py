from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer.binary_activation import binary_activation, custom_activation
from util.distribution_tools import get_mean_logvar, split_node

class VaeEncoderGeco(tf.keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, middle_dim, bbn_dim, cbn_dim):
        """

        :param middle_dim: hidden units
        :param bbn_dim: binary bottleneck size
        :param cbn_dim: continuous bottleneck size
        """
        super(VaeEncoderGeco, self).__init__()
        self.code_length = bbn_dim
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='gelu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim*2,kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim*2,kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))

        self.reconstruction1 = tf.keras.layers.Dense(2048, activation='gelu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.reconstruction2 = tf.keras.layers.Dense(4096, activation='sigmoid',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))

    def call(self, inputs, training=True, **kwargs):
        batch_size = tf.shape(inputs)[0]
        fc_1 = self.fc_1(inputs)

        mean, logvar = get_mean_logvar(self, inputs)
        if training:
          self.eps = tf.clip_by_value(tf.random.normal(shape=mean.shape),-5,5)
        else:
          self.eps = tf.zeros(shape=mean.shape)
        bbn =  custom_activation(mean,logvar,self.eps)

        cbn = self.fc_2_2(fc_1)
        mean2, logvar2 = split_node(cbn)
        if training:
          self.eps2 = tf.clip_by_value(tf.random.normal(shape=mean2.shape),-5,5)
        else:
          self.eps2 = tf.zeros(shape=mean2.shape)
        
        cbn = mean2 + logvar2*self.eps2
        return bbn, cbn


# noinspection PyAbstractClass
class Decoder(tf.keras.layers.Layer):
    def __init__(self, middle_dim, feat_dim):
        """
        :param middle_dim: hidden units
        :param feat_dim: data dim
        """
        super(Decoder, self).__init__()
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='gelu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2 = tf.keras.layers.Dense(feat_dim, activation='gelu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))

    def call(self, inputs, **kwargs):
        fc_1 = self.fc_1(inputs)
        return self.fc_2(fc_1)




if __name__ == '__main__':
    a = tf.ones([2, 4096], dtype=tf.float32)
    encoder = Encoder(1024, 64, 512)
    b = encoder(a)
    print(encoder.trainable_variables)
