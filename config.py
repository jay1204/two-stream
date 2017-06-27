import numpy as np 
from easydict import EasyDict as ed 

config = ed()

config.NUM_CLASSES = 101
config.TRAIN = ed()
config.TRAIN.BATCH_SIZE = 256
config.TRAIN.EPOCH = 1000
config.TRAIN.DROP_OUT = 1.0

default = ed()

default.data_root = 'data/'
default.data_list_path = default.data_root + 'ucfTrainTestlist/'
default.train_list = default.data_list_path + 'trainlist01.txt'
default.test_list = default.data_list_path + 'testlist01.txt'
default.label_list = default.data_list_path + 'classInd.txt'
default.data_dir = default.data_root + 'jpegs_256/'
default.train_lst = default.data_list_path + 'train.lst'
default.valid_lst = default.data_list_path + 'valid.lst'
default.test_lst = default.data_list_path + 'test.lst'

nn = ed()

nn.model_dir = 'pretrained_model/'
nn.resnet.pretrained_model_name = 'resnet-50'
nn.resnet.pretrained_model_epoch = 0
nn.resnet.url_prefix = 'http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50'






