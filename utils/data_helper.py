import os
import sys
import random
import numpy as np
import csv
from ..logger import logger
from ..config import ucf_img


def get_UCF_image_lst(regenerate=False, train_ratio=1.0):
    """
    Given the training list and testing list file, make .lsts file for them
    """
    if not os.path.exists(ucf_img.train_lst) or os.path.exists(ucf_img.valid_lst) or regenerate:
        make_image_lst(input_file_path=ucf_img.train_list, data_dir=ucf_img.data_dir,
                       label_file=ucf_img.label_list, train_ratio=train_ratio, file_type='train', )
        logger.info('create train.lst and valid.lst')
    if not os.path.exists(ucf_img.test_lst) or regenerate:
        make_image_lst(input_file_path=ucf_img.test_list, data_dir= ucf_img.data_dir,
                       label_file=ucf_img.label_list, train_ratio=train_ratio, file_type='test')
        logger.info('create test.lst')

    return


def make_image_lst(input_file_path, data_dir, label_file, train_ratio, file_type):
    data_file_list, input_file_label_names = read_input_file(input_file_path, data_dir)
    label_dict = read_label_file(label_file)
    
    # given labelNames and label_dict, retrieve labels for each item data_file_list
    labels = map(lambda x: label_dict[x], input_file_label_names)
    write_to_file_wrapper(input_file_path, data_file_list, labels, train_ratio, file_type)
    return


def write_to_file_wrapper(input_file_path, data_file_list, labels, train_ratio, file_type):
    if file_type == 'train':
        train_file = extract_dir(input_file_path) + '/train.lst'
        valid_file = extract_dir(input_file_path) + '/valid.lst'
        train_indices, valid_indices = train_valid_split(len(data_file_list), train_ratio=train_ratio)
        write_to_file(train_file, train_indices, data_file_list, labels)
        write_to_file(valid_file, valid_indices, data_file_list, labels)
    elif file_type == 'test':
        test_file = extract_dir(input_file_path) + '/test.lst'
        test_indices = range(len(data_file_list))
        write_to_file(test_file, test_indices, data_file_list, labels)
    return


def write_to_file(file_name, indices, data_file_list, labels):
    file_handler = csv.writer(open(file_name, "w"), delimiter='\t', lineterminator='\n')
    image_list = []
    counter = 0
    for i in indices:
        for img in os.listdir(data_file_list[i]):
            if img.endswith('jpg'):
                image_list.append((counter, labels[i], data_file_list[i] + '/' + img))
                counter += 1
    
    for il in image_list:
        file_handler.writerow(il)
    return 


def train_valid_split(num, train_ratio=0.8):
    """
    Given a split rate, rate of all nums is training indices, rest is valid indices
    """
    train_size = int(num * train_ratio)
    train_indices = sorted(np.random.choice(num, size = train_size, replace = False).tolist())
    valid_indices = []
    train_index = 0
    for i in xrange(num):
        if train_index >= len(train_indices) or i != train_indices[train_index]:
            valid_indices.append(i)
        else:
            train_index += 1
    return train_indices, valid_indices


def extract_dir(input_file_path):
    """
    Get the directory of the inputFile, 
    for example:'ucf/list01.txt'=> 'ucf'
    """
    return '/'.join((input_file_path.split('/')[:-1]))


def read_label_file(label_file):
    """
    read the label document and return a dict{VideoName: label}
    The data format in the labelFile is like:
    '2 ApplyLipstick'
    """
    label_file = open(label_file, 'rU')
    label_dict = {}
    for line in label_file:
        line_list = line.replace('\n', '').split(' ')
        label_dict[line_list[1]] = int(line_list[0])
    return label_dict


def read_input_file(input_file_path, data_dir):
    """
    The data format in the input_file should be like:
    'ApplyEyeMakeup/v_ApplyEyeMakeup_g10_c02.avi 1'
    """
    input_file = open(input_file_path, 'rU')
    input_file_info = filter(lambda x: x.find('.avi') != -1, input_file)
    # process conflict in InputFileInfo
    input_file_info = map(lambda x: x.replace('HandStandPushups', 'HandstandPushups'), input_file_info)
    data_file_list = map(lambda x: data_dir + (x.split('.')[0]).split('/')[-1], input_file_info)
    input_file_label_names = map(lambda x: (x.split('.')[0]).split('/')[-2], input_file_info)

    return data_file_list, input_file_label_names
