import numpy as np 
import mxnet as mx 
from mxnet.executor_manager import _split_input_slice
import mxnet.ndarray as nd
from mxnet.image import *


class ImageIter(mx.io.DataIter):
    """
    This class is a wrapper of the basic mx.io.DataIter. 
    it reads raw image files

    """
    def __init__(self, batch_size, data_shape, path_img_list, ctx=None, shuffle=False, data_name='data',
                 label_name='label', work_load_list=None, **kwargs):
        super(ImageIter, self).__init__()

        self.batch_size = batch_size

        with open(path_img_list) as fin:
            img_list = {}
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                label = np.array([float(i) for i in line[1:-1]])
                key = int(line[0])
                img_list[key] = (label, line[-1])

        self.pre_process = []
        for key in kwargs.keys():
            if kwargs[key]:
                self.pre_process.append(key)
        #CreateAugmenter(data_shape, **kwargs)

        self.img_list = img_list
        self.shuffle = shuffle
        self.img_size = len(img_list.keys())
        self.seq = np.arange(self.img_size, dtype = np.int)

        self.data_shape = data_shape
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]

        self.data_name = data_name
        self.label_name = label_name
        self.work_load_list = work_load_list

        self.provide_data = [(data_name, (batch_size, ) + data_shape)]
        self.provide_label = [(label_name, (batch_size, ))]

        self.cur = 0
        self.reset()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.seq)

    def iter_next(self):
        return self.cur + self.batch_size <= self.img_size

    def next(self):
        if self.iter_next():
            batch_data, batch_label = self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch([batch_data], [batch_label])

    def get_batch(self):
        batch_start = self.cur
        batch_end = min(self.cur + self.batch_size, self.img_size)
        batch_indices = self.seq[batch_start:batch_end]

        #work_load_list = self.work_load_list
        #if work_load_list is None:
        #    work_load_list = [1] * len(self.ctx)

        #slices = _split_input_slice(self.batch_size, work_load_list)

        #data_list = []
        #label_list = []
        #for each_slice in slices:
        #    imgs_list = map(lambda x: self.imglist[batch_indices[x]], range(each_slice.start, each_slice.stop))
        #    data, label = self.read_imgs(imgs = imgs_list)
        #    data_list.append(data)
        #    label_list.append(label)
        imgs_list = map(lambda x: self.img_list[x], batch_indices)
        batch_data, batch_label = self.read_imgs(imgs_list)
        return batch_data, batch_label

    def read_imgs(self, imgs_list):
        """
        Given a list of img_path, read those images and preprocess them to fit into the required data_shape 
        Inputs:
            - imgs: a list of tuple(img_path, label)
            - preprocess: a dict indicating how to preprocess image
        """
        c, h, w = self.data_shape
        batch_data = nd.empty((self.batch_size, c, h, w))
        batch_label = nd.empty(self.batch_size)
        self.images = []
        self.img_paths = []
        for i, img in enumerate(imgs_list):
            label, img_path = img
            image = self.next_image(img_path)
            if image.shape != self.data_shape:
                raise AssertionError('The size of the image is not matched with the required data_shape!')
            batch_data[i][:] = image
            batch_label[i][:] = label

        return batch_data, batch_label

    def pre_process_image(self, image):
        """Transforms input data with specified augmentation."""
        for process in self.pre_process:
            if process == 'rand_crop':
                c, h, w = self.data_shape
                image, _ = random_crop(image, (h, w))
        return image

    def next_image(self, img_path):
        image = self.load_one_image(img_path)
        image = self.pre_process_image(image)
        image = self.post_process_image(image)

        return image

    @staticmethod
    def load_one_image(img_path):
        with open(img_path, 'rb') as fp:
            image_info = fp.read()

        return mx.img.imdecode(image_info)

    @staticmethod
    def post_process_image(image):
        """
        Transform the image to make it shape as (channel, height, width)
        """
        return nd.transpose(image, axes=(2, 0, 1))







