import sys
import time
import datetime
from PIL import Image
import os
import numpy as np

def imsave(img, filename, mode='L'):
    img = np.clip(img, 0, 255)
    img = np.asarray(img, dtype='int8')
    img = Image.fromarray(img, mode=mode)

    if os.path.exists('{}.png'.format(filename)):
        i = 1
        while os.path.exists('{}({:d}).png'.format(filename, i)):
            i += 1
        filename = '{}({:d}).png'.format(filename, i)
    else:
        filename = '{}.png'.format(filename)

    img.save(filename)


def unzip(l):
    # http://stackoverflow.com/a/12974504
    return [list(t) for t in zip(*l)]

class Timer(object):
    """A class that wraps some useful time related methods"""
    def __init__(self):
        self.markers = {}
    def setMarker(self, marker_name):
        self.markers[marker_name] = time.time()
    def timeSince(self, marker_name):
        t = int(time.time() - self.markers[marker_name])
        return str(datetime.timedelta(seconds=t))
    def sleep(self, t):
        time.sleep(t)

class Printer(object):
    """A class for printing text animations."""
    def __init__(self, wait_time):
        self.wait_time = wait_time
        self.t = time.time()
        self.last_length = 0
    def overwrite(self, message='', wait=True):
        """Prints message without starting a new line."""
        if time.time() - self.t >= self.wait_time or not wait:
            sys.stdout.write('\r' + ' '*self.last_length + '\r')
            sys.stdout.flush()
            sys.stdout.write(message)
            sys.stdout.flush()
            self.t = time.time()
            self.last_length = len(message)
    def clear(self):
        """Clears whatever was last written using the method overwrite without starting a new line."""
        self.overwrite(wait=False)

import numpy as np
class OneHotEncoder(object):
    def __init__(self, number_of_categories):
        self.number_of_categories = number_of_categories

    def encode(self, categories):
        """
        Encodes a category or list of categories into a one-hot array or a list of one-hot arrays
        :param category: an integer or list of integers representing categories.
        """
        if isinstance(categories, int):
            a = np.zeros(self.number_of_categories)
            a[categories] = 1
        else:
            a = np.zeros((len(categories), self.number_of_categories))
            a[np.arange(a.shape[0]), categories.astype('int')] = 1
        return a

    def decode(self, onehot):
        """
        Decodes a one hot array or list of one hot arrayes into an integer or list of integers
        :param onehot: a list of integers or list of a list of integers
        """
        # ensure that onehot is of shape (n, m)
        onehot = np.asarray(onehot)
        return np.argmax(onehot, axis=1)

    def maxval(self, onehot):
        onehot = np.asarray(onehot)
        return np.amax(onehot, axis=len(onehot.shape) - 1)
