# deepimagehashing-VAE

This is the main repository for our improvement on [Auto-Encoding Twin-Bottleneck Hashing](https://arxiv.org/abs/2002.11930). 
Note that the codebase is also an adapted version of the [TBH repository](https://github.com/ymcidence/TBH).

## Requirements
```angular2
python=3.6
tensorflow>=2.5
numpy
matplotlib
```

## Data
This work supports `tf.data.TFRecordDataset` as the data feed. 
We use the cifar10 dataset provided by the TBH authors:
* Cifar-10 ([Training](https://drive.google.com/open?id=1Ie0ucwA1r5tG9pETWbYaR50Y2Mz76h0A), [Test](https://drive.google.com/open?id=1GdHaetvz6cwo2UE7_epMFci62ViNiDjB))

For other datasets, please refer to [`util/data/make_data.py`](./util/data/make_data.py) to build TFRecords.

Please organize the data folder as follows:
```angular2
data
  |-cifar10 (or other dataset names)
    |-train.tfrecords
    |-test.tfrecords
```

Simply run
```angular2
python ./run_tbh.py
```
to train the model.

The resulting checkpoints will be placed in `./result/set_name/model/date_of_today` with tensorboard events in `./result/set_name/log/date_of_today`.

The mAP results shown on tensorboard are just for illustration (the actual score would be slightly higher than the ones on tensorboard), 
since I do not update all dataset codes upon testing. Please kindly evaluate the results by saving the proceeded codes after training.


The visualisations can be recreated by running
```angular2
python ./vis.py
```
