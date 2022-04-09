import numpy as np

class ReLu:

    def __init__(self):
        self.last_input = None

    def relu(self, x):
        if len(x.shape) > 1:
            self.last_input = x
        else:
            self.last_input = x.reshape(1,x.shape[0])
        return np.maximum(0, x)

    def relu_grad(self, x):
        dZ = np.array(x, copy=True)  # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well.
        dZ[self.last_input <= 0] = 0
        return dZ