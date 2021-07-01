import tensorflow as tf
import numpy as np

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return -.5 * ((tf.exp(-logvar/2)*(sample - mean)) ** 2. + logvar + log2pi)


def get_mean_logvar(enc, x):
  fc_1 = enc.fc_1(x)
  bbn = enc.fc_2_1(fc_1)
  mean, logvar = split_node(bbn)
  return mean, logvar

def split_node(bbn):
  mean, logvar = tf.split(bbn, num_or_size_splits=2, axis=1)
  if(mean.shape[0] == None):
    mean = tf.reshape(mean,[1,mean.shape[1]])
    logvar = tf.reshape(logvar,[1,logvar.shape[1]])
  return mean, logvar



def elbo_decomposition(mean, logvar, eps, actor_loss):
    logpx = -actor_loss
    qz_samples = eps * tf.exp(logvar * .5) + mean
    nlogpz = tf.reduce_mean(log_normal_pdf(qz_samples, 0., 0.),axis=0)
    nlogqz_condx = tf.reduce_mean(log_normal_pdf(qz_samples, mean, logvar),axis=0)
    marginal_entropies, joint_entropy = estimate_entropies(qz_samples, mean, logvar)

    # Independence term
    # KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
    dependence = (- joint_entropy + tf.math.reduce_mean(marginal_entropies))[0]

    # Information term
    # KL(q(z|x)||q(z)) = log q(z|x) - log q(z)
    information = (- tf.math.reduce_mean(nlogqz_condx) + joint_entropy)[0]

    # Dimension-wise KL term
    # sum_j KL(q(z_j)||p(z_j)) = sum_j (log q(z_j) - log p(z_j))
    dimwise_kl = tf.math.reduce_mean(- marginal_entropies + nlogpz)

    # Compute sum of terms analytically
    # KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
    analytical_cond_kl = tf.reduce_mean(- nlogqz_condx + nlogpz)

    return logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy


def estimate_entropies(qz_samples, mean, logvar, n_samples=1024, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).
    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)
    Inputs:
    -------
        qz_samples (N, K) Variable
        mean (N, K) Variable
        logvar (N, K) Variable
    """

    # S batch size, K hidden units

    S, K = qz_samples.shape

    
    weights = -tf.math.log(float(S))

    marginal_entropies = tf.zeros(K)
    joint_entropy = tf.zeros(1)

    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = log_normal_pdf(qz_samples[k:k + batch_size,:],mean[k:k + batch_size,:], logvar[k:k + batch_size,:])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        marginal_entropies += - (weights + log_sum_exp(logqz_i , dim=0, keepdim=False))
        # computes - log q(z) summed over minibatch
        logqz = tf.math.reduce_mean(logqz_i, axis=1)  # (N, S)
        joint_entropy += (tf.math.log(float(S)) - log_sum_exp(logqz, dim=0, keepdim=False))

    marginal_entropies /= S
    joint_entropy /= S
    return marginal_entropies, joint_entropy

def calc_mi(model, x):
  """Approximate the mutual information between x and z
  I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
  Returns: Float"""
  
  # [x_batch, nz]
  mu, logvar = get_mean_logvar(model.encoder, x)

  batch_size, nz = mu.shape

  # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
  neg_entropy = tf.math.reduce_mean(-0.5 * nz * tf.math.log(2 * np.pi)- 0.5 * tf.math.reduce_sum(1 + logvar,axis=-1))

  # [z_batch, 1, nz]
  std = tf.math.exp(tf.math.scalar_mul(0.5,logvar))

  mu_expd = tf.expand_dims(mu,1)
  std_expd = tf.expand_dims(std,1)

  eps = tf.random.normal(std_expd.shape)

  z_samples = mu_expd + tf.math.multiply(eps, std_expd)

  # [1, x_batch, nz]
  mu, logvar = tf.expand_dims(mu,0), tf.expand_dims(logvar,0)
  var = tf.math.exp(logvar)

  # (z_batch, x_batch, nz)
  dev = z_samples - mu
  
  # (z_batch, x_batch) tf.reduce_sum((dev ** 2)/var,axis=-1)
  log_density = -0.5 * tf.math.multiply(tf.reduce_sum(1/var,axis=-1), tf.reduce_sum(dev ** 2,axis=-1)) - \
      0.5 * (nz * tf.math.log(2 * np.pi) + tf.reduce_sum(logvar,axis=-1))

  # log q(z): aggregate posterior
  # [z_batch]
  log_qz = log_sum_exp(log_density, dim=1) - tf.math.log(float(batch_size))
  #print("qz",tf.math.reduce_any(tf.math.is_inf(log_qz)))

  return (neg_entropy - tf.math.reduce_mean(log_qz,axis=-1)).numpy()
  
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m = tf.math.reduce_max(value, axis=dim, keepdims=True)
        value0 = value - m
        if keepdim is False:
            m = tf.squeeze(m,dim)
        return m + tf.math.log(tf.math.reduce_sum(tf.math.exp(value0), axis=dim, keepdims=keepdim))
    else:
        m = tf.math.reduce_max(value)
        sum_exp = tf.math.reduce_sum(tf.math.exp(value - m))
        return m + tf.math.log(sum_exp)