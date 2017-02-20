import os
import json
import time
import cPickle as pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

class Transformer(object):
    def __init__(self, mean=[128, 128, 128], shape=[227,227]):
        self.mean = np.array(mean, dtype=np.float32)
        self.shape=shape

    def preprocess(self, im):
        im = scipy.misc.imresize(im, self.shape) # resize
        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)


class NTBDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1]
        )
        top[1].reshape(self.batch_size, params['num_labels'])

        print_info("NTBDataLayer", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            while True:
                try:
                    im, multilabel = self.batch_loader.load_next_image()
                    break
                except ValueError as e:
                    print e

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.ntb_root = params['ntb_root']
        self.num_labels = params['num_labels']
        # parse images metadata
        metada_file_path = os.path.join(
            self.ntb_root, 'nets', params['sub_dir'], 'data', params['split'] + '.pickle'
        )
        with open(metada_file_path) as f:
            self.metadata = pickle.load(f)
        self.indexlist = self.metadata.keys()
        shuffle(self.indexlist)
        self._cur = 0  # current image
        self.epoch = 0
        self.transformer = Transformer(shape=params['im_shape'])

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist)
        )

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            print("Epoch {} finished".format(self.epoch))
            self.epoch += 1
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]
        image_path = os.path.join(
            self.ntb_root, self.metadata[index]['folder'], index + '.jpg'
        )
        im = np.asarray(Image.open(image_path))

        # do a simple horizontal flip as data augmentation
        #flip = np.random.choice(2)*2-1
        #im = im[:, ::flip, :]

        # Load and prepare ground truth
        multilabel = np.zeros(self.num_labels).astype(np.float32)
        for label in self.metadata[index]['labels']:
            multilabel[label] = 1

        self._cur += 1
        try:
            return self.transformer.preprocess(im), multilabel
        except Exception as e:
            raise ValueError("Error during processing {}: {}".format(str(index), str(e)))


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'ntb_root', 'im_shape', 'num_labels']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    print "{} initialized for sub_dir: {}, split: {}, with bs: {}, im_shape: {}, num_labels: {}.".format(
        name,
        params['sub_dir'],
        params['split'],
        params['batch_size'],
        params['im_shape'],
        params['num_labels'],
    )