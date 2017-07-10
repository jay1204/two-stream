import numpy as np 
import mxnet as mx 

class ImageIter(mx.io.DataIter):
    """
    This class is a wrapper of the basic mx.io.DataIter. 
    it reads raw image files
        - 
    """
    def __init__(self, batch_size, data_shape, path_imglist, path_root, ctx, shuffle=False, work_load_list = None, **kwargs):
        super(ImageIter, self).__init__()

        self.batch_size = batch_size

        with open('path_imglist') as fin:
            imglist = {}
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                label = np.array([float(i) for i in line[1:-1]])
                key = int(line[0])
                imglist[key] = (label, line[-1])

        self.preprocess = kwargs

        self.imglist = imglist 
        self.shuffle = shuffle
        self.img_size = len(imglist.keys())
        self.seq = self.arange(self.imgSize)
        self.resize = resize

        self.data_shape = data_shape
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]

        self.data_name = 'data'
        self.label_name = 'label'
        self.work_load_list = work_load_list

        self.cur = 0
        self.reset()
        self.next()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            self.seq = np.random.shuffle(self.seq)


    def iter_next(self)：
        return self.cur + self.batch_size <= self.img_size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return 

    def get_batch(self):
        batch_start = self.cur
        batch_end = min(self.cur + self.batch_size, self.img_size)
        batch_indices = self.seq[batch_start:batch_end]

        work_load_list = self.work_load_list
        if work_load_list is None:
            work_load_list = [1] * len(self.ctx)

        slices = _split_input_slice(self.batch_size, work_load_list)

        for each_slice  in slices:
            imgs_list = map(lambda x: self.imglist[batch_indices[x]], range(each_slice.start, each_slice.stop))
            data, label = self.read_imgs(imgs = imgs_list, data_shape = self.data_shape, preprocess = self.preprocess)

    def read_imgs(self, imgs, data_shape, preprocess):
        """
        Given a list of img_path, read those images and preprocess them to fit into the required data_shape 
        Inputs:
            - imgs: a list of tuple(img_path, label)
            - preprocess: a dict indicating how to preprocess image
        """
        self.images = []
        self.img_paths = []
        for img in imgs:
            img_path, label = img
            image = self.load_one_image(img_path)
            self.images.append(image)
            self.img_paths.append(img_path)


    def load_one_image(self, img_path):
        with open(framePath, 'rb') as fp:
            imageInfo = fp.read()

        return mx.img.imdecode(imageInfo)






