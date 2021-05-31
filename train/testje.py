import sys
if True:
  sys.path.append("Google Drive/Mastersthesis/TBH")
else:
  sys.path.append("gdrive/My Drive/Mastersthesis/TBH")
from train import tbh_train
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tbh_train.train("cifar10",32,512,1024)