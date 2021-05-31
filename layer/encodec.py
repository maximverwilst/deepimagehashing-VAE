from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer.binary_activation import binary_activation, custom_activation
from util.distribution_tools import get_mean_logvar, split_node


class VaeEncoderSingle(tf.keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, middle_dim, bbn_dim, cbn_dim):
        """

        :param middle_dim: hidden units
        :param bbn_dim: binary bottleneck size
        :param cbn_dim: continuous bottleneck size
        """
        super(VaeEncoder, self).__init__()
        self.code_length = bbn_dim
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim*2,kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim, activation='sigmoid')

    def call(self, inputs, training=True, continuous=True, **kwargs):
        batch_size = tf.shape(inputs)[0]
        fc_1 = self.fc_1(inputs)

        mean, logvar = get_mean_logvar(self, inputs)
        if training:
          self.eps = tf.clip_by_value(tf.random.normal(shape=mean.shape),-5,5)
        else:
          self.eps = tf.zeros(shape=mean.shape)
        
        bbn =  custom_activation(mean,logvar,self.eps)

        cbn = self.fc_2_2(fc_1)
        if not(continuous):
          tf.zeros(cbn.shape)
        return bbn, cbn

class VaeEncoder(tf.keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, middle_dim, bbn_dim, cbn_dim):
        """

        :param middle_dim: hidden units
        :param bbn_dim: binary bottleneck size
        :param cbn_dim: continuous bottleneck size
        """
        super(VaeEncoder, self).__init__()
        self.code_length = bbn_dim
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim*2,kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim, activation='sigmoid')

    def call(self, inputs, training=True, continuous=True, **kwargs):
        batch_size = tf.shape(inputs)[0]
        fc_1 = self.fc_1(inputs)

        mean, logvar = get_mean_logvar(self, inputs)
        if training:
          self.eps = tf.clip_by_value(tf.random.normal(shape=mean.shape),-5,5)
        else:
          self.eps = tf.zeros(shape=mean.shape)
        
        bbn =  custom_activation(mean,logvar,self.eps)

        cbn = self.fc_2_2(fc_1)
        """mean2, logvar2 = split_node(cbn)
        if training:
          self.eps2 = tf.clip_by_value(tf.random.normal(shape=mean2.shape),-5,5)
        else:
          self.eps2 = tf.zeros(shape=mean2.shape)
        
        cbn = mean2 + logvar2*self.eps2"""
        return bbn, cbn

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
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='gelu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=10), bias_constraint=tf.keras.constraints.MaxNorm(max_value=10))
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim*2,kernel_constraint=tf.keras.constraints.MaxNorm(max_value=10), bias_constraint=tf.keras.constraints.MaxNorm(max_value=10))
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim*2)# removed sigmoid activation function

        self.reconstruction1 = tf.keras.layers.Dense(1024, activation='gelu')
        self.reconstruction2 = tf.keras.layers.Dense(2048, activation='sigmoid')
        #self.reconstruction1 = tf.keras.layers.Dense(2048, activation='gelu')
        #self.reconstruction2 = tf.keras.layers.Dense(4096, activation='sigmoid')

    def call(self, inputs, training=True, continuous=True, **kwargs):
        batch_size = tf.shape(inputs)[0]
        fc_1 = self.fc_1(inputs)

        
        """
        bbn = self.fc_2_1(fc_1)
        self.eps = tf.ones([batch_size, self.code_length]) / 2.
        bbn, _ = binary_activation(bbn, self.eps)
        """
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

class VaeEncoderGecoRaw(tf.keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, middle_dim, bbn_dim, cbn_dim):
        """

        :param middle_dim: hidden units
        :param bbn_dim: binary bottleneck size
        :param cbn_dim: continuous bottleneck size
        """
        super(VaeEncoderGecoRaw, self).__init__()
        self.code_length = bbn_dim

        #conv encoder
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Reshape((100,100,1), input_shape=(100*100,)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))

        model.add(tf.keras.layers.Flatten())
        #model.add(tf.keras.layers.Dense(4096))
        model.add(tf.keras.layers.Dense(middle_dim, activation='relu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1)))

        self.fc_1 = model
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim*2,kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim*2, activation='sigmoid',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))

        model3 = tf.keras.models.Sequential()
        model3.add(tf.keras.layers.Reshape((100,100,1), input_shape=(100*100,)))
        model3.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model3.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model3.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model3.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model3.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model3.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model3.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model3.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model3.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model3.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model3.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))

        model3.add(tf.keras.layers.Flatten())

        self.reconstruction1 = model3

        model2 = tf.keras.models.Sequential()
        model2.add(tf.keras.layers.Dense(512, input_shape=(512,), activation = 'relu'))
        model2.add(tf.keras.layers.Reshape((2,2,128)))
        model2.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.UpSampling2D((2, 2)))
        model2.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.UpSampling2D((2, 2)))
        model2.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.UpSampling2D((2, 2)))
        model2.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.UpSampling2D((2, 2)))
        model2.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.Flatten())
        self.reconstruction2 = model2

    def call(self, inputs, training=True, continuous=True, **kwargs):
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


class Encoder(tf.keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, middle_dim, bbn_dim, cbn_dim):
        """

        :param middle_dim: hidden units
        :param bbn_dim: binary bottleneck size
        :param cbn_dim: continuous bottleneck size
        """
        super(Encoder, self).__init__()
        self.code_length = bbn_dim
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu')
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim)
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim, activation='sigmoid')

    def call(self, inputs, training=True, **kwargs):
        batch_size = tf.shape(inputs)[0]
        fc_1 = self.fc_1(inputs)

        bbn = self.fc_2_1(fc_1)
        #if training:
        #  self.eps = tf.random.uniform([batch_size, self.code_length], maxval=1)
        #else:
        self.eps = tf.ones([batch_size, self.code_length]) / 2.

        bbn, _ = binary_activation(bbn, self.eps)
        cbn = self.fc_2_2(fc_1)
        return bbn, cbn


# noinspection PyAbstractClass
class Decoder(tf.keras.layers.Layer):
    def __init__(self, middle_dim, feat_dim):
        """
        :param middle_dim: hidden units
        :param feat_dim: data dim
        """
        super(Decoder, self).__init__()
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='gelu')
        self.fc_2 = tf.keras.layers.Dense(feat_dim, activation='gelu')

    def call(self, inputs, **kwargs):
        fc_1 = self.fc_1(inputs)
        return self.fc_2(fc_1)

# noinspection PyAbstractClass
class DecoderRaw(tf.keras.layers.Layer):
    def __init__(self, middle_dim, feat_dim):
        """
        :param middle_dim: hidden units
        :param feat_dim: data dim
        """
        super(DecoderRaw, self).__init__()

        #conv decoder
        model2 = tf.keras.models.Sequential()
        model2.add(tf.keras.layers.Dense(512, input_shape=(1024,), activation = 'relu'))
        model2.add(tf.keras.layers.Reshape((2,2,128)))
        model2.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.UpSampling2D((2, 2)))
        model2.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.UpSampling2D((2, 2)))
        model2.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.UpSampling2D((2, 2)))
        model2.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.UpSampling2D((2, 2)))
        model2.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='relu'))
        model2.add(tf.keras.layers.Flatten())

        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu',kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), bias_constraint=tf.keras.constraints.MaxNorm(max_value=1))
        self.fc_2 = model2

    def call(self, inputs, **kwargs):
        fc_1 = self.fc_1(inputs)
        return self.fc_2(fc_1)



if __name__ == '__main__':
    a = tf.ones([2, 4096], dtype=tf.float32)
    encoder = Encoder(1024, 64, 512)
    b = encoder(a)
    print(encoder.trainable_variables)
