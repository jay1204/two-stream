import numpy as np 
import mxnet as mx 

class ImageIter(io.DataIter):
	"""
	This class is a wrapper of the basic mx.io.DataIter. 
	it reads raw image files
		- 
	"""
	def __init__(self, batch_size, data_shape, path_imglist, path_root, random_crop = False, , shuffle=False):
		super(ImageIter, self).__init__()

		self.batch_size = batch_size

		with open('path_imglist') as fin:
			imglist = {}
            imgkeys = []
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                label = nd.array([float(i) for i in line[1:-1]])
                key = int(line[0])
                imglist[key] = (label, line[-1])

        print imglist
