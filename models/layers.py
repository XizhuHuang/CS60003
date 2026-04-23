import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros(output_dim)

        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, dout):
        X = self.cache
        self.dW = X.T @ dout / X.shape[0]
        self.db = np.sum(dout, axis=0) / X.shape[0]
        return dout @ self.W.T
    
    def backward_input(self, dout):
        """
        dout: (N, out_dim)
        return dX: (N, in_dim)
        """
        return dout @ self.W.T