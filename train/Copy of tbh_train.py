from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model.tbh import TBH
from util.data.dataset import Dataset
from util.eval_tools import eval_cls_map
from util.distribution_tools import log_normal_pdf, calc_mi, get_mean_logvar, elbo_decomposition, split_node
from util.optimizers.cocob import COCOB
from layer.twin_bottleneck import build_adjacency_hamming
import matplotlib.pyplot as plt
from meta import REPO_PATH
import os
from time import gmtime, strftime

import numpy as np


def hook(query, base, label_q, label_b, at=1000):
    return eval_cls_map(query, base, label_q, label_b, at)


@tf.function
def adv_loss(real, fake):
    real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real), real))
    fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake), fake))
    total_loss = real_loss + fake_loss
    return total_loss


@tf.function
def reconstruction_loss(pred, origin):
    return  tf.reduce_mean(tf.reduce_sum(tf.math.square(pred - origin),axis=1))
    #return tf.reduce_mean(tf.nn.l2_loss(pred - origin))

@tf.function
def divergent_loss(mean, logvar, eps):
  """
  z = mean + logvar*eps
  logpz = tf.reduce_mean(log_normal_pdf(z, 0., 0.),axis = 1)
  logqz_x = tf.reduce_mean(log_normal_pdf(z, mean, logvar),axis = 1)
  return -tf.reduce_mean(logpz - logqz_x)
  """
  logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = elbo_decomposition(mean, logvar, eps, 0.)
  return -(information + dimwise_kl + dependence)
  """
  kl_loss = 1 + logvar - tf.math.square(mean) - tf.math.exp(logvar)
  kl_loss = tf.math.reduce_sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  kl_loss = tf.math.reduce_mean(kl_loss)
  return kl_loss
  """
  


def IWAE_KL(x, reconstruction_mu, latent_mu, latent_logsigma, z, tol):
    
    log_p = tf.math.reduce_sum(tf.math.pow(reconstruction_mu - x, 2), axis = -1) - tol
    log_q = tf.math.reduce_sum(-0.5 * ((z - latent_mu)/tf.math.exp(latent_logsigma))**2 - latent_logsigma, axis=-1)
    log_prior = tf.math.reduce_sum(-0.5 * tf.math.pow(z,2), -1)
    
    total = log_p/1000. + log_prior - log_q
    total = total - tf.math.reduce_max(total)

    weights = tf.math.exp(total)
    
    normalized_weights = weights / tf.stop_gradient(tf.math.reduce_sum(weights))
    out = -tf.math.reduce_mean(tf.math.reduce_sum(normalized_weights * total, 0))
    
    return out

def train_step(model: TBH, batch_data, bbn_dim, cbn_dim, batch_size, actor_opt: tf.optimizers.Optimizer,
               critic_opt: tf.optimizers.Optimizer, divergence_opt: tf.optimizers.Optimizer, lambd, aggressive=False, binarytraining=False):
    random_binary = (tf.sign(tf.random.uniform([batch_size, bbn_dim]) - 0.5) + 1) / 2
    random_cont = tf.random.uniform([batch_size, cbn_dim])


    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape, tf.GradientTape() as divergence_tape, tf.GradientTape() as divergence_tape2:
        model_input = [batch_data, random_binary, random_cont]
        model_output = model(model_input, training=True, continuous=True)

        #critic_loss = adv_loss(model_output[5], model_output[3])# + adv_loss(model_output[4], model_output[2])
        #actor_loss = reconstruction_loss(model_output[1], batch_data[1]) - \
        #             adv_loss(model_output[5], model_output[3])# - \
                     #adv_loss(model_output[4], model_output[2])

        
        mean, logvar = get_mean_logvar(model.encoder,batch_data[1])
        eps = model.encoder.eps
        divergence_loss = divergent_loss(mean, logvar, eps)#/max(lambd*1e6,1)
        
       
        #mean2, logvar2 = split_node(cbn)
        #eps2 = model.encoder.eps2
        #divergence_loss2 = divergent_loss(mean2, logvar2, eps2)
        
        fc_1 = model.encoder.fc_1(batch_data[1])
        cbn = model.encoder.fc_2_2(fc_1)
        latent_mu, latent_logsigma = split_node(cbn)

        z = latent_mu + latent_logsigma*model.encoder.eps2
        
        fc_2 = model.encoder.reconstruction1(model_output[1])
        reconstruction_mu = model.encoder.reconstruction2(fc_2)
        tol = 2350
        constraint = tf.reduce_mean(tf.reduce_sum(tf.math.pow(reconstruction_mu - batch_data[1], 2), axis = 1) - tol)

        KL_div = IWAE_KL(batch_data[1], reconstruction_mu, latent_mu, latent_logsigma, z, tol)

        loss = KL_div + constraint*lambd
        
        
        actor_scope = model.encoder.trainable_variables + model.decoder.trainable_variables + model.tbn.trainable_variables

        divergence_scope = model.encoder.fc_2_1.trainable_variables+model.encoder.fc_1.trainable_variables
        #divergence_scope2 = model.encoder.fc_2_2.trainable_variables + model.encoder.fc_1.trainable_variables
        #critic_scope = model.dis_2.trainable_variables# + model.dis_1.trainable_variables
        
        #if binarytraining:
        #actor_scope = model.encoder.fc_1.trainable_variables + model.encoder.fc_2_1.trainable_variables + model.tbn.trainable_variables + \
        #                  model.decoder.trainable_variables
        #else:
        #    actor_scope = model.encoder.fc_2_2.trainable_variables + model.tbn.trainable_variables
        #if aggressive:
        #    actor_scope = model.encoder.fc_2_1.trainable_variables

        actor_gradient = actor_tape.gradient(loss, sources=actor_scope)#!loss
        divergence_gradient = divergence_tape.gradient(divergence_loss, sources=divergence_scope)
        #divergence_gradient2 = divergence_tape2.gradient(divergence_loss2, sources=divergence_scope2)
        #critic_gradient = critic_tape.gradient(critic_loss, sources=critic_scope)
        
        #if not(binarytraining):
        #divergence_opt.apply_gradients(zip(divergence_gradient, divergence_scope))
        #  divergence_opt.apply_gradients(zip(divergence_gradient2, divergence_scope2))
        actor_opt.apply_gradients(zip(actor_gradient, actor_scope))
        #critic_opt.apply_gradients(zip(critic_gradient, critic_scope))
          
    #constraint+ tol
    #return model_output[0].numpy(), actor_loss.numpy(), critic_loss.numpy(), divergence_loss.numpy(), 0
    return model_output[0].numpy(), constraint + tol, 0, divergence_loss.numpy(), constraint#divergence_loss2#.numpy()
                                                       

def test_step(model: TBH, batch_data):
    model_input = [batch_data]
    model_output = model(model_input, training=False)
    return model_output.numpy()


def train(set_name, bbn_dim, cbn_dim, batch_size, middle_dim=1024, max_iter=1000000, continue_path=False, continue_step=0, warmup = False):
    print("check")
    model = TBH(set_name, bbn_dim, cbn_dim, middle_dim)
    if continue_path != False:
      model2 = tf.keras.models.load_model(continue_path)
    
    data = Dataset(set_name=set_name, batch_size=batch_size, code_length=bbn_dim)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=5000,
        decay_rate=0.65)
    lr_schedule2 = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=5000,
        decay_rate=0.65)
    
    actor_opt = tf.keras.optimizers.Adam(1e-4)#COCOB(alpha=10000)
    critic_opt = tf.keras.optimizers.Adam(1e-4)#COCOB(alpha=10000)
    divergence_opt = tf.keras.optimizers.Adam(1e-8)#COCOB(alpha=10000)

    train_iter = iter(data.train_data)
    test_iter = iter(data.test_data)

    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    result_path = os.path.join(REPO_PATH, 'result', set_name)
    save_path = os.path.join(result_path, 'model')
    summary_path = os.path.join(result_path, 'log', time_string)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(actor_opt=actor_opt, critic_opt=critic_opt, divergence_opt=divergence_opt, model=model)
    
    save_name = os.path.join(save_path)
    manager = tf.train.CheckpointManager(checkpoint, save_name, max_to_keep=1)

    binarytraining = False
    best_hook = 0
    lambd = 1.
    constrain_ma = 1.
    alpha = .99
    for i in range(continue_step, max_iter):
      with writer.as_default():
          train_batch = next(train_iter)

          train_code, actor_loss, critic_loss, divergence_loss, constraint = train_step(model, train_batch, bbn_dim, cbn_dim, batch_size, actor_opt, critic_opt, divergence_opt, lambd, binarytraining = binarytraining)
          train_label = train_batch[2].numpy()
          train_entry = train_batch[0].numpy()
          data.update(train_entry, train_code, train_label, 'train')
          
          if i == 0:
            constrain_ma = constraint
          else:
            constrain_ma = alpha * constrain_ma + (1. - alpha) * constraint
          if i % 100 == 0:
            lambd *= tf.clip_by_value(tf.math.exp(constrain_ma), .9, 1.1)
            lambd = tf.clip_by_value(lambd,1e-6,1e6)

          
          if (i + 1) % 100 == 0:

              test_batch = next(test_iter)
              train_hook = hook(train_code, train_code, train_label, train_label, at=min(batch_size, 1000))
              """mean, logvar = get_mean_logvar(model.encoder,test_batch[1])
              eps = model.encoder.eps
              logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = elbo_decomposition(mean, logvar, eps,-actor_loss)

              print('Estimated  ELBO: {}, Dependence: {}, Information: {}, Dimension-wise KL: {}, Analytical E_p(x)[ KL(q(z|x)||p(z)) ]: {}'.format(logpx - analytical_cond_kl, dependence, information, dimwise_kl, analytical_cond_kl))

              tf.summary.scalar('test/elbo', logpx - analytical_cond_kl, step=i)
              tf.summary.scalar('test/dependence', dependence, step=i)
              tf.summary.scalar('test/information', information, step=i)
              tf.summary.scalar('test/dimwise_kl', dimwise_kl, step=i)
              tf.summary.scalar('test/analytical kl', analytical_cond_kl, step=i)"""

              tf.summary.scalar("train/lambd", lambd, step=i)
              tf.summary.scalar("train/constrain", constrain_ma, step=i)
              
              tf.summary.scalar('train/actor', actor_loss, step=i)
              tf.summary.scalar('train/critic', critic_loss, step=i)
              tf.summary.scalar('train/divergence', divergence_loss, step=i)
              tf.summary.scalar('train/hook', train_hook, step=i)
              writer.flush()
              print('batch {}: train_hook {}, actor {}, critic {}, divergence {}, lambda {}'.format(i, train_hook, actor_loss, critic_loss, divergence_loss, lambd))#, cur_mi))

          if (i + 1) % 2000 == 0:
              print('Testing!!!!!!!!')
              test_batch = next(test_iter)
              test_code = test_step(model, test_batch)
              test_label = test_batch[2].numpy()
              test_entry = test_batch[0].numpy()
              data.update(test_entry, test_code, test_label, 'test')
              test_hook = hook(test_code, data.train_code, test_label, data.train_label, at=1000)
              tf.summary.scalar('test/hook', test_hook, step=i)
              if test_hook >= best_hook:
              #if actor_loss <= best_hook:
                #best_hook = actor_loss
                best_hook = test_hook
                tf.keras.models.save_model(model, filepath = save_path)
              print("test_hook: ", test_hook)


if __name__ == '__main__':
    train('cifar10', 32, 512, 400)
