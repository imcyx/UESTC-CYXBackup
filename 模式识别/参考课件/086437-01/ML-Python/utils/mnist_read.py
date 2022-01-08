# -*- coding: utf-8 -*-
import numpy as np
import struct
import os


def load_mnist(path, train=True):
    """
    加载mnist文件
    """
    data_type = 'train'
    if not train:
        data_type = 't10k'
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % data_type)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % data_type)
    with open(labels_path, 'rb') as lb:
        magic, n = struct.unpack('>II', lb.read(8))
        labels = np.fromfile(lb, dtype=np.uint8)
    with open(images_path, 'rb') as img:
        magic, num, rows, cols = struct.unpack('>IIII', img.read(16))
        images = np.fromfile(img, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

"""
import matplotlib.pyplot as plt

path = '../data/mnist/'
train_images, train_labels = load_mnist(path)
test_images, test_labels = load_mnist(path, False)
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(30):
    images = np.reshape(train_images[i], [28, 28])
    ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(train_labels[i]))
plt.show()
"""