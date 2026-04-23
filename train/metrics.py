import numpy as np

def accuracy(logits, y):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def confusion_matrix(logits, y, num_classes):
    preds = np.argmax(logits, axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y, preds):
        cm[t, p] += 1
    
    return cm