import numpy as np 
from easydict import EasyDict as ed 

config = ed()

config.num_classes = 101

config.train_img = ed()
config.train_img.batch_size = 60
config.train_img.epoch = 1
config.train_img.drop_out = 1.0

config.train_flow = ed()
config.train_flow.batch_size = 80
config.train_flow.epoch = 1
config.train_flow.drop_out = 1.0

ucf_img = ed()

ucf_img.data_root = 'data/'
ucf_img.data_list_path = ucf_img.data_root + 'ucfTrainTestlist/'
ucf_img.train_list = ucf_img.data_list_path + 'trainlist01.txt'
ucf_img.test_list = ucf_img.data_list_path + 'testlist01.txt'
ucf_img.label_list = ucf_img.data_list_path + 'classInd.txt'
ucf_img.data_dir = ucf_img.data_root + 'jpegs_256/'
ucf_img.train_lst = ucf_img.data_list_path + 'train.lst'
ucf_img.valid_lst = ucf_img.data_list_path + 'valid.lst'
ucf_img.test_lst = ucf_img.data_list_path + 'test.lst'

ucf_flow = ed()


nn = ed()
nn.resnet = ed()
nn.resnet.model_dir = 'pretrained_model/'
nn.resnet.pretrained_model_name = 'resnet-50'
nn.resnet.pretrained_model_epoch = 0
nn.resnet.url_prefix = 'http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50'
nn.resnet.data_shape = (3, 224, 224)








