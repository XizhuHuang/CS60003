from models.layers import LinearLayer
from models.activations import Activation
from models.loss import SoftmaxCrossEntropy
import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation='relu'):
        self.linear1 = LinearLayer(input_dim, hidden_dim1)
        self.act1 = Activation(activation)

        self.linear2 = LinearLayer(hidden_dim1, hidden_dim2)
        self.act2 = Activation(activation)

        self.linear3 = LinearLayer(hidden_dim2, output_dim)

        self.layers = [
            self.linear1, self.act1,
            self.linear2, self.act2,
            self.linear3
        ]

        self.loss = SoftmaxCrossEntropy()

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        dout = self.loss.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def backward_input(self, dlogits):
        """
        只对输入反传，不更新参数
        """

        dout = dlogits

        for layer in reversed(self.layers):
            dout = layer.backward_input(dout)

        return dout

    def get_regularization_loss(self, reg):
        loss = 0
        for layer in [self.linear1, self.linear2, self.linear3]:
            loss += 0.5 * reg * np.sum(layer.W ** 2)
        return loss

    def save_weights(self, path):
        np.savez(path,
            W1=self.linear1.W, b1=self.linear1.b,
            W2=self.linear2.W, b2=self.linear2.b,
            W3=self.linear3.W, b3=self.linear3.b
        )

    def load_weights(self, path):
        data = np.load(path)
        self.linear1.W = data['W1']
        self.linear1.b = data['b1']
        self.linear2.W = data['W2']
        self.linear2.b = data['b2']
        self.linear3.W = data['W3']
        self.linear3.b = data['b3']