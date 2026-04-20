import numpy as np


class RunningNormalizer:
    def __init__(self, size):
        self.mean = np.zeros(size)
        self.var = np.ones(size)
        self.count = 1e-4
        
    def update(self, x):
        batch_mean = x.mean(axis=0) if x.ndim > 1 else x
        batch_var = x.var(axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total) / total
        self.count = total
        
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)