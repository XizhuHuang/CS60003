import numpy as np

class SoftmaxCrossEntropy:
    def forward(self, X, y):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1 - eps)
        loss = -np.log(probs_clipped[np.arange(len(y)), y]).mean()
        self.cache = (probs, y)
        return loss

    def backward(self):
        probs, y = self.cache
        dX = probs.copy()
        dX[np.arange(len(y)), y] -= 1
        return dX / len(y)