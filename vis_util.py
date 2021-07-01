import matplotlib.pyplot as plt
from train import tbh_train
from model.tbh import TBH
from util.eval_tools import eval_cls_map, gen_sim_mat, compute_hamming_dist
from util.distribution_tools import get_mean_logvar
import tensorflow as tf
from util.data.dataset import Dataset
import numpy as np
import os
import sys
from meta import REPO_PATH

#generate train codes
def generate_train(model, dataset, data_parser, set="cifar10"):
  record_name = os.path.join(REPO_PATH, 'data', set, "train" + '.tfrecords')
  data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50)
  data = data.batch(50000)

  trainiter = iter(data)
  train = next(trainiter)

  feat_in = tf.cast(train[1], dtype=tf.float32)
  mean, logvar = get_mean_logvar(model.encoder, feat_in)

  bbntrain = (tf.sign(mean) + 1.0) / 2.0
  dataset.update(train[0].numpy(), bbntrain.numpy(), train[2].numpy(), 'train')
  return dataset

#generate test codes + compute mAP
def generate_test(model, dataset, data_parser, set="cifar10"):
  record_name = os.path.join(REPO_PATH, 'data', set, "test" + '.tfrecords')
  scores = []
  data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50).batch(10000)
  testiter = iter(data)
  test = next(testiter)

  feat_in = tf.cast(test[1], dtype=tf.float32)
  mean, logvar = get_mean_logvar(model.encoder, feat_in)

  bbntest = (tf.sign(mean) + 1.0) / 2.0
  dataset.update(test[0].numpy(), bbntest.numpy(), test[2].numpy(), 'test')
  return dataset

def test_hook(loaded_model, dataset, data_parser):
  record_name = os.path.join(REPO_PATH, 'data', dataset.set_name, "test" + '.tfrecords')
  scores = []
  for i in range(10):
    data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50).shuffle(1000).batch(1000)

    testiter = iter(data)
    test = next(testiter)

    feat_in = tf.cast(test[1], dtype=tf.float32)
    mean, logvar = get_mean_logvar(loaded_model.encoder, feat_in)
    bbntest = (tf.sign(mean) + 1.0) / 2.0
    dataset.update(test[0].numpy(), bbntest.numpy(), test[2].numpy(), 'test')

    test_hook = eval_cls_map(bbntest.numpy(), dataset.train_code, test[2].numpy(), dataset.train_label, at=1000)
    scores.append(test_hook)
  return scores

#calculate precision-recall curve values
def get_prec_rec_matrix(dataset, data_parser, model):
  record_name = os.path.join(REPO_PATH, 'data', "cifar10", "test" + '.tfrecords')
  data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50).shuffle(1000).batch(1000)
  testiter = iter(data)
  test = next(testiter)
  feat_in = tf.cast(test[1], dtype=tf.float32)
  mean, logvar = get_mean_logvar(model.encoder, feat_in)
  bbntest = (tf.sign(mean) + 1.0) / 2.0
  dataset.update(test[0].numpy(), bbntest.numpy(), test[2].numpy(), 'test')

  query, target, cls1, cls2, at = bbntest.numpy(), dataset.train_code, test[2].numpy(), dataset.train_label, 50000

  top_k = at

  sim_mat = gen_sim_mat(cls1, cls2)
  query_size = query.shape[0]
  distances = compute_hamming_dist(query, target)
  dist_argsort = np.argsort(distances)

  prec_rec = [[0 for i in range(top_k)] for i in range(query_size)]
  map_count = 0.
  average_precision = 0.
  average_recall = 0.
  for i in range(query_size):
      gt_count = 0.
      precision = 0.
      top_k = at if at is not None else dist_argsort.shape[1]
      for j in range(top_k):
          this_ind = dist_argsort[i, j]
          if sim_mat[i, this_ind] == 1:
              prec_rec[i][j] = 1
              gt_count += 1.
              precision += gt_count / (j + 1.)
      average_recall += gt_count/5000
      if gt_count > 0:
          average_precision += precision / gt_count
          map_count += 1.
  average_recall /= (query_size)
  prec_rec = np.array(prec_rec)

  avg_prec = [0 for i in range(100)]
  avg_rec = [0 for i in range(100)]
  for t in range(1,101):
    map_count = 0.
    for i in range(prec_rec.shape[0]):
      gt_count = np.sum(prec_rec[i][:int(prec_rec.shape[1]*t/100)])
      prec = float(gt_count) / (prec_rec.shape[1]*t/100)
      if gt_count>0:
        map_count += 1
        avg_prec[t-1] += prec
      avg_rec[t-1] += gt_count/5000


    avg_prec[t-1] /= prec_rec.shape[0]
    avg_rec[t-1] /= prec_rec.shape[0]
  return avg_prec, avg_rec
  
def top_10_retrieval(model,dataset, data_parser,orig_train,orig_test):
  record_name = os.path.join(REPO_PATH, 'data', dataset.set_name, "test" + '.tfrecords')
  data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50).batch(10).shuffle(1000)

  testiter = iter(data)
  test = next(testiter)
  test = next(testiter)

  feat_in = tf.cast(test[1], dtype=tf.float32)
  mean, logvar = get_mean_logvar(model.encoder, feat_in)
  bbntest = (tf.sign(mean) + 1.0) / 2.0

  query, target, cls1, cls2 = bbntest.numpy(), dataset.train_code, test[2].numpy(), dataset.train_label
  sim_mat = gen_sim_mat(cls1, cls2)
  query_size = query.shape[0]
  distances = compute_hamming_dist(query, target)
  dist_argsort = np.argsort(distances)

  retrievals = [0 for i in range(10)]
  for i in range(10):
    retrievals[i] = orig_train[0][dist_argsort[i][0:10]]
    retrievals[i] = np.concatenate(retrievals[i],axis=1)
    retrievals[i] = np.concatenate((orig_test[0][test[0][i]],np.zeros([32,5,3], dtype = int),retrievals[i]),axis=1)
  retrievals = np.concatenate(retrievals,axis = 0)

  for i in range(10):
    for j in range(10):
      
       if (orig_train[1][dist_argsort[i][0:10]][j] == orig_test[1][test[0][i]]):
         retrievals[i*32:i*32+1,j*32+37:j*32+69,:3] = np.concatenate([np.zeros([1,32,1],dtype=int),np.ones([1,32,1],dtype=int)*130,np.zeros([1,32,1],dtype=int)],axis=2)
         retrievals[i*32+31:i*32+32,j*32+37:j*32+69,:3] = np.concatenate([np.zeros([1,32,1],dtype=int),np.ones([1,32,1],dtype=int)*130,np.zeros([1,32,1],dtype=int)],axis=2)
         retrievals[i*32:i*32+32,j*32+37:j*32+38,:3] = np.concatenate([np.zeros([32,1,1],dtype=int),np.ones([32,1,1],dtype=int)*130,np.zeros([32,1,1],dtype=int)],axis=2)
         retrievals[i*32:i*32+32,j*32+68:j*32+69,:3] = np.concatenate([np.zeros([32,1,1],dtype=int),np.ones([32,1,1],dtype=int)*130,np.zeros([32,1,1],dtype=int)],axis=2)
       else:
         retrievals[i*32:i*32+1,j*32+37:j*32+69,:3] = np.concatenate([200*np.ones([1,32,1],dtype=int),np.ones([1,32,1],dtype=int),np.zeros([1,32,1],dtype=int)],axis=2)
         retrievals[i*32+31:i*32+32,j*32+37:j*32+69,:3] = np.concatenate([200*np.ones([1,32,1],dtype=int),np.ones([1,32,1],dtype=int),np.zeros([1,32,1],dtype=int)],axis=2)
         retrievals[i*32:i*32+32,j*32+37:j*32+38,:3] = np.concatenate([200*np.ones([32,1,1],dtype=int),np.ones([32,1,1],dtype=int),np.zeros([32,1,1],dtype=int)],axis=2)
         retrievals[i*32:i*32+32,j*32+68:j*32+69,:3] = np.concatenate([200*np.ones([32,1,1],dtype=int),np.ones([32,1,1],dtype=int),np.zeros([32,1,1],dtype=int)],axis=2)

        
  return retrievals