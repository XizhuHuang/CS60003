import numpy as np
from models.layers import LinearLayer

class SGDMomentum:
    def __init__(self, layers, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.linear_layers = [l for l in layers if isinstance(l, LinearLayer)]

        self.velocities = []
        for layer in self.linear_layers:
            self.velocities.append({
                'W': np.zeros_like(layer.W),
                'b': np.zeros_like(layer.b)
            })

    def step(self):
        for i, layer in enumerate(self.linear_layers):
            vW = self.momentum * self.velocities[i]['W'] + self.lr * (layer.dW + self.weight_decay * layer.W)
            vb = self.momentum * self.velocities[i]['b'] + self.lr * layer.db

            layer.W -= vW
            layer.b -= vb

            self.velocities[i]['W'] = vW
            self.velocities[i]['b'] = vb