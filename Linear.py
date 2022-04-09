import numpy as np

class Linear:

    def __init__(self, input_dim, output_dim):

        # setting conv parameters
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
        self.last_input = None

    def forward(self, input):
        self.last_input = input

        return np.dot(input,self.W) + self.bias

    def backprop(self, err, lr):
        #m = self.last_input.shape[1]
        w_grad = lr * np.dot(self.last_input.T, err)
        b_grad = lr * np.average(err, axis=0)
        err2 = np.dot(err, self.W.T)

        self.W = self.W - w_grad
        self.bias = self.bias - b_grad

        return err2