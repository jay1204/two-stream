import mxnet as mx
from ..utils import get_UCF_image_lst, get_model, load_pretrained_model,refactor_model
from ..config import nn, ucf_img, config
from ..logger import logger
from ..classes import ImageIter
from easydict import EasyDict as ed


class CNN(object):
    """
    This class takes a pre-trained CNN model which has been trained on ImageNet, and train the model using our data set.
    """
    def __init__(self, pretrained_model, dataset, ctx, train_setting, num_classes, prefix):
        """
        Initialize the parameters need for our network

        :param pretrained_model: a dict of the pre-trained network setting
        :param dataset: a dict of train, valid and test data info
        """
        self.dataset = dataset
        self.pretrained_model = pretrained_model
        self.ctx = ctx
        self.train = train_setting
        self.num_classes = num_classes
        self.prefix = prefix

    def configure_pretrained_model(self):
        # load pre-trained model
        sym, arg_params, _ = load_pretrained_model(self.pretrained_model.pretrained_model_name,
                                                   self.pretrained_model.pretrained_model_epoch,
                                                   self.pretrained_model.model_dir, ctx=self.ctx)

        # replace the last layer with a fully connected layer
        new_sym, new_args = refactor_model(sym, arg_params, self.num_classes)
        return new_sym, new_args

    def train_model(self, learning_rate, shuffle=True, label_name='softmax_label', rand_crop=True):
        """
        Write the training pipeline
        :return:
        """
        logger.info("Start training CNN model")

        train_iter = ImageIter(batch_size=self.train.batch_size, data_shape=self.pretrained_model.data_shape,
                               path_img_lst=self.dataset.train_lst, ctx=self.ctx, shuffle=shuffle,
                               label_name=label_name, work_load_list=None, rand_crop=rand_crop)
        valid_iter = ImageIter(batch_size=self.train.batch_size * 10, data_shape=self.pretrained_model.data_shape,
                               path_img_lst=self.dataset.valid_lst, ctx=self.ctx, shuffle=shuffle,
                               label_name=label_name, work_load_list=None, rand_crop=rand_crop)

        net, args = self.configure_pretrained_model()
        mod = mx.mod.Module(symbol=net, context=self.ctx)
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))

        # mod.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', 0.5), ))
        lr_sch = mx.lr_scheduler.FactorScheduler(step=100, factor=0.9)
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', learning_rate),
                                                              ('lr_scheduler', lr_sch)))

        metric = mx.metric.create('acc')
        count = 0
        train_acc = []
        valid_acc = []
        valid_accuracy = 0.0
        for epoch in range(self.train.epoch):
            train_iter.reset()
            valid_iter.reset()
            metric.reset()
            for batch in train_iter:
                logger.info("The batch {}".format(batch))
                mod.forward(batch, is_train=True)
                mod.update_metric(metric, batch.label)
                mod.backward()
                mod.update()
                count += 1
                train_acc.append(metric.get())
                if count % 100 == 0:
                    # logger.info("The train accuracy of the %d-th iteration is %f"%(count, train_acc[-1][1]))
                    print "The train accuracy of the %d-th iteration is %f"%(count, train_acc[-1][1])
                    score = mod.score(valid_iter.next(), ['acc'])
                    valid_acc.append(score)
                    print "The valid accuracy of the %-th iteration is %f"%(count, valid_acc[-1][1])
                    # logger.info("The valid accuracy of the %-th iteration is %f"%(count, valid_acc[-1][1]))
                    if valid_acc[-1] > valid_accuracy:
                        valid_accuracy = valid_acc[-1]
                        mod.save(self.prefix, count)

                if count >= 100 and train_acc[-1][1] - train_acc[-2][1] <= 0.0001:
                    break

        return train_acc, valid_acc
