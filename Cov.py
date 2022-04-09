import numpy as np

class Conv:

    def __init__(self, num_filters, channel, kernel_size, stride):

        # setting conv parameters

        self.kernel_size = kernel_size
        self.stride = stride
        self.num_filters = num_filters
        self.filters = np.random.randn(self.num_filters, channel, self.kernel_size, self.kernel_size)/9
        self.last_input = None
        self.col = None
        self.col_W = None


    def cov_forward(self, x):

        self.last_input = x

        FN, C, FH, FW = self.filters.shape
        N, C, H, W = x.shape
        h = 1 + int((H - FH) // self.stride)
        w = 1 + int((W - FW) // self.stride)
        weight = self.filters.reshape(FN, -1).T

        vector = np.zeros((h, w, N, C, FH, FW))

        for i in range(h):  # loop over vertical axis of the output volume
            for j in range(w):  # loop over horizontal axis of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    i_start = i * self.stride
                    i_end = i * self.stride + self.kernel_size
                    j_start = j * self.stride
                    j_end = j * self.stride + self.kernel_size

                    vector[i, j, :, :, :, :] = x[:, :, i_start:i_end, j_start:j_end]

        vector = vector.transpose(2, 0, 1, 3, 4, 5).reshape(N * h * w, -1)

        out = np.dot(vector, weight)
        out = out.reshape(N, h, w, -1).transpose(0, 3, 1, 2)

        self.col = vector
        self.col_W = weight

        return out


    def cov_backward(self, err, lr):

        FN, C_after, FH, FW = self.filters.shape
        N, C_before, H, W = self.last_input.shape
        h = 1 + int((H - FH) // self.stride)
        w = 1 + int((W - FW) // self.stride)

        dout = err.transpose(0, 2, 3, 1).reshape(-1, FN)
        dW = np.dot(self.col.T, dout)
        dW = dW.transpose(1, 0).reshape(FN, C_after, FH, FW)

        # dx

        dx = np.zeros(self.last_input.shape)

        for n in range(N):
            for c in range(C_after):
                for i in range(h):  # loop over vertical axis of the output volume
                    for j in range(w):  # loop over horizontal axis of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        i_start = i
                        i_end = i + self.kernel_size
                        j_start = j
                        j_end = j + self.kernel_size

                        dx[n, :, i_start:i_end, j_start:j_end] += self.filters[c, :, :, :] * err[n, c, i, j]

        # update filters
        self.filters -= lr * dW

        return dx








