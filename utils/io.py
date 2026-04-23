import numpy as np

def save_model(model, path):
    model.save_weights(path)

def load_model(model, path):
    model.load_weights(path)
    return model