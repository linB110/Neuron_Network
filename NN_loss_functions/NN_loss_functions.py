import math as m
import numpy as np

class loss_function:
    def __init__(self, input, target):
        self.input = input
        self.target = target
        self.loss = None
        
    # use BCE as an example
    def compute_loss(self):
        y_pred = self.input
        y_true = self.target
        
        samples = len(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)

        probability = y_pred_clipped[range(samples), y_true]
        
        loss = -np.mean(np.log(probability))
        return loss
    
vec = np.array([[0.7, 0.1, 0.2], 
                [0.1, 0.5, 0.4],
                [0.02, 0.9, 0.08]])
lbl = [0, 1, 1]

lsf = loss_function(vec, lbl)
loss = lsf.compute_loss()

print(loss)
        
