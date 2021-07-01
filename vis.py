import vis_util
import tensorflow as tf
from util.data.dataset import Dataset
from meta import REPO_PATH
from util.data.set_processor import SET_DIM, SET_LABEL, SET_SPLIT, SET_SIZE
import matplotlib.pyplot as plt

path = "\\result\\cifar10\\32bit\\model"
loaded_model = tf.keras.models.load_model(REPO_PATH + path)

set_name = "cifar10"
def data_parser(tf_example: tf.train.Example):
            feat_dict = {'id': tf.io.FixedLenFeature([], tf.int64),
                         'feat': tf.io.FixedLenFeature([SET_DIM.get(set_name, 4096)], tf.float32),
                         'label': tf.io.FixedLenFeature([SET_LABEL.get(set_name, 10)], tf.float32)}
            features = tf.io.parse_single_example(tf_example, features=feat_dict)

            _id = tf.cast(features['id'], tf.int32)
            _feat = tf.cast(features['feat'], tf.float32)
            _label = tf.cast(features['label'], tf.int32)
            return _id, _feat, _label

dataset = Dataset(set_name = set_name, batch_size=1024, code_length=32)
train_dataset = vis_util.generate_train(loaded_model, dataset, data_parser)

# test mAP for 1000 random queries (10 times)
print(vis_util.test_hook(loaded_model, train_dataset, data_parser))

# calculate precision-recall curves
avg_prec, avg_rec = vis_util.get_prec_rec_matrix(train_dataset, data_parser, loaded_model)
plt.plot(avg_rec, avg_prec)
plt.show()

# visualise top-10 retrievals
orig_train, orig_test = tf.keras.datasets.cifar10.load_data()
retrievals = vis_util.top_10_retrieval(loaded_model, train_dataset, data_parser, orig_train, orig_test)

fig, ax = plt.subplots(figsize=(18, 20))
ax.imshow(retrievals)
plt.show()
