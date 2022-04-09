from math import log2

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from Cov import *
from Linear import *
from MaxPool import *
from ReLu import *
import mnist
import random



class NN:

    # calculate cross entropy

    def softmax_probs(self, out):
        e_x = np.exp(out - np.max(out, axis=1, keepdims=True))
        probs = e_x / np.sum(e_x, axis=1, keepdims=True)
        return probs

    def cross_E(self, probs, y_true):  # CE
        N = probs.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y_true])) / N

        # CE derivative
        dx = probs.copy()
        dx[np.arange(N), y_true] -= 1
        dx /= N

        return loss, dx

    def loss_acc(self, out, label):
        prob = self.softmax_probs(out)
        error, _ = self.cross_E(prob, label)

        # get predictions
        predictions = np.argmax(prob, axis=1)

        # get accuracy
        accuracy = np.mean(predictions == label)

        print(f"loss:{error} accuracy:{accuracy}")






if __name__ == '__main__':
    random.seed(1)
    train_images = mnist.train_images()[:1024]
    train_labels = mnist.train_labels()[:1024]
    print(train_images.shape, train_labels.shape)

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

    for k in range(10):
        for i in range(0, 1024, batch_size):
            #
            #print(i)
            train = train_images[i:i+batch_size].reshape(train_images[i:i+batch_size].shape[0],1,train_images[i:i+batch_size].shape[1],train_images[i:i+batch_size].shape[2])
            labels = train_labels[i:i+batch_size]

            x = conv1.cov_forward(train)
            x = relu1.relu(x)
            x = pool1.forward(x)
            x = conv2.cov_forward(x)
            x = relu2.relu(x)
            x = pool2.forward(x)
            x = fc1.forward(x.flatten().reshape(batch_size, -1))
            x = relu3.relu(x)
            x = fc2.forward(x)



            error, gradient = nn.cross_E(nn.softmax_probs(x), labels)
            print(error)


            gradient = fc2.backprop(gradient, 1e-3)
            gradient = relu3.relu_grad(gradient)
            gradient = fc1.backprop(gradient, 1e-3)
            gradient = pool2.pool_backward(gradient)
            gradient = relu2.relu_grad(gradient)
            gradient = conv2.cov_backward(gradient, 1e-3)
            gradient = pool1.pool_backward(gradient)
            gradient = relu1.relu_grad(gradient)
            gradient = conv1.cov_backward(gradient, 1e-3)


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

        print("Saving weights")
        with open('conv1.npy', 'wb') as f:
            np.save(f, conv1.filters)

        with open('conv2.npy', 'wb') as f:
            np.save(f, conv2.filters)

        with open('fc1_W.npy', 'wb') as f:
            np.save(f, fc1.W)

        with open('fc1_b.npy', 'wb') as f:
            np.save(f, fc1.bias)


        with open('fc2_W.npy', 'wb') as f:
            np.save(f, fc2.W)

        with open('fc2_b.npy', 'wb') as f:
            np.save(f, fc2.bias)



