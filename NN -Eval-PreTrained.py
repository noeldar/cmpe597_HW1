from math import log2

import numpy as np
from NN import *
from Cov import *
from Linear import *
from MaxPool import *
from ReLu import *
import mnist
import random


if __name__ == '__main__':
    test_images = mnist.test_images()[:1024]
    test_labels = mnist.test_labels()[:1024]
    print(test_images.shape, test_labels.shape)

    batch_size = 64

    nn = NN()
    conv1 = Conv(4, 1, 5, 1)
    relu1 = ReLu()
    pool1 = MaxPool()
    conv2 = Conv(8, 4, 5, 1)
    relu2 = ReLu()
    pool2 = MaxPool()
    fc1 = Linear(2592, 128)
    relu3 = ReLu()
    fc2 = Linear(128, 10)

    with open('conv1.npy', 'rb') as f:
        conv1.filters = np.load(f)

    with open('conv2.npy', 'rb') as f:
        conv2.filters = np.load(f)

    with open('fc1_W.npy', 'rb') as f:
        fc1.W = np.load(f)

    with open('fc1_b.npy', 'rb') as f:
        fc1.bias = np.load(f)

    with open('fc2_W.npy', 'rb') as f:
        fc2.W = np.load(f)

    with open('fc2_b.npy', 'rb') as f:
        fc2.bias = np.load(f)

    for i in range(0, 1024, batch_size):
            test = test_images[i:i+batch_size].reshape(test_images[i:i+batch_size].shape[0],1,test_images[i:i+batch_size].shape[1],test_images[i:i+batch_size].shape[2])

            x = conv1.cov_forward(test)
            x = relu1.relu(x)
            x = pool1.forward(x)
            x = conv2.cov_forward(x)
            x = relu2.relu(x)
            x = pool2.forward(x)
            x = fc1.forward(x.flatten().reshape(batch_size, -1))
            x = relu3.relu(x)
            x = fc2.forward(x)

            if i == 0:
                pt = x
            else:
                pt = np.concatenate((pt, x), axis=0)



    nn.loss_acc(pt, test_labels)


