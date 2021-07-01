import scipy.io as sio
import numpy as np
import os

SET_SPLIT = ['train', 'test']
SET_DIM = {'cifar10': 4096, "NETosis": 20000, "multimodal":400,"coco":2048,"2018_02_27_P103_evHeLa_4M":1280,"2018_03_16_P103_shPerk_bQ":1280}
SET_LABEL = {'cifar10': 10, "NETosis": 2, "multimodal":1,"coco":80,"2018_02_27_P103_evHeLa_4M":1,"2018_03_16_P103_shPerk_bQ":1}
SET_SIZE = {'cifar10': [50000, 10000], "NETosis": [27881,5577],"2018_03_16_P103_shPerk_bQ": [92577,10399], "multimodal":[1910982,213676], "coco":[10000,5000],"2018_02_27_P103_evHeLa_4M":[24948,2772]}

def cifar_processor(root_folder):
    class_num = 10

    def reader(file_name, part=SET_SPLIT[0]):
        data_mat = sio.loadmat(file_name)
        feat = data_mat[part + '_data']
        label = np.squeeze(data_mat[part + '_label'])
        fid = np.arange(0, feat.shape[0])
        label = np.eye(class_num)[label]

        return {'feat': feat, 'label': label, 'fid': fid}

    train_name = os.path.join(root_folder, 'cifar10_fc7_train.mat')
    train_dict = reader(train_name)
    test_name = os.path.join(root_folder, 'cifar10_fc7_test.mat')
    test_dict = reader(test_name, part=SET_SPLIT[1])

    return train_dict, test_dict



SET_PROCESSOR = {'cifar10': cifar_processor, "NETosis": cifar_processor}
