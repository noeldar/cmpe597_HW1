import numpy as np

class MaxPool:

    def __init__(self):
        self.last_input = None
        #self.indexes = []

    def forward(self, x):
        self.last_input = x
        N, C, H, W = x.shape

        h = 1 + (H - 2) // 1
        w = 1 + (W - 2) // 1
        output = np.zeros((N, C, h, w))

        for n in range(N):
            for c in range(C):
                for i in range(h):
                    for j in range(w):
                        output[n, c, i, j] = np.max(x[n, c, i:i + 2, j:j + 2].reshape(-1))
                        #a,b = np.unravel_index(np.argmax(x[n, c, i:i + 2, j:j + 2], axis=None), x[n, c, i:i + 2, j:j + 2].shape)
                        #self.indexes.append((n,c,i+a,j+b,i,j))

        return output#, self.indexes

    def max_pool_backward(self, dout):
        x = self.last_input
        pool_height = 2
        pool_width = 2
        stride = 1
        N, C, H, W = x.shape
        _, _, out_H, out_W = dout.reshape(N, C, 1 + (H - pool_height) // stride, 1 + (W - pool_width) // stride).shape

        dx = np.zeros_like(x)

        for i in range(N):
            curr_dout = dout[i, :].reshape(C, out_H * out_W)
            c = 0
            for j in range(0, H - pool_height + 1, stride):
                for k in range(0, W - pool_width + 1, stride):
                    curr_region = x[i, :, j:j + pool_height, k:k + pool_width].reshape(C, pool_height * pool_width)
                    curr_max_idx = np.argmax(curr_region, axis=1)
                    #print(curr_max_idx)
                    curr_dout_region = curr_dout[:, c]
                    curr_dpooling = np.zeros_like(curr_region)
                    curr_dpooling[np.arange(C), curr_max_idx] = curr_dout_region
                    dx[i, :, j:j + pool_height, k:k + pool_height] = curr_dpooling.reshape(C, pool_height, pool_width)
                    c += 1

        return dx

    def pool_filter(self, x):

        mask = x == np.max(x)

        return mask


    def pool_backward(self, dout):
        x = self.last_input
        pool_height = 2
        pool_width = 2
        stride = 1
        N, C, H, W = x.shape
        dout = dout.reshape(N, C, 1 + (H - pool_height) // stride, 1 + (W - pool_width) // stride)
        _, _, h, w = dout.shape

        dx = np.zeros(x.shape)

        for n in range(N):  # loop over the training examples
            # select training example from A_prev (≈1 line)
            for c in range(C):  # loop over the channels (depth)
                for i in range(h):  # loop on the vertical axis
                    for j in range(w):  # loop on the horizontal axis

                            # Find the corners of the current "slice" (≈4 lines)
                            i_start = i
                            i_end = i_start + 2
                            j_start = j
                            j_end = j_start + 2

                            prev_slice = x[n, c, i_start:i_end, j_start:j_end]
                            mask = self.pool_filter(prev_slice)
                            dx[n, c, i_start:i_end, j_start:j_end] += np.multiply(mask, dout[n, c, i, j])
                            
        return dx


