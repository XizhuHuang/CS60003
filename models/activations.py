import numpy as np

class Activation:
    def __init__(self, activation='relu'):
        self.activation = activation
        self.cache = None

    def forward(self, X):
        if self.activation == 'relu':
            out = np.maximum(0, X)
        elif self.activation == 'sigmoid':
            out = np.where(X >= 0, 
                   1 / (1 + np.exp(-X)), 
                   np.exp(X) / (1 + np.exp(X)))
        elif self.activation == 'tanh':
            out = np.tanh(X)
        else:
            raise ValueError("Unsupported activation")
        
        self.cache = out
        return out

    def backward(self, dout):
        out = self.cache

        if self.activation == 'relu':
            grad = (out > 0).astype(float)
        elif self.activation == 'sigmoid':
            grad = out * (1 - out)
        elif self.activation == 'tanh':
            grad = 1 - out**2

        return dout * grad
    
    def backward_input(self, dout):
        """
        dout: (N, D)
        return dX: (N, D)
        """
        out = self.cache

        if self.activation == 'relu':
            grad = (out > 0).astype(float)
        elif self.activation == 'sigmoid':
            grad = out * (1 - out)
        elif self.activation == 'tanh':
            grad = 1 - out**2

        return dout * grad