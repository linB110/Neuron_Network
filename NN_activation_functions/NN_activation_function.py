import numpy as np

class activation_functions:
    def __init__(self, input):
        self.x = input
    
    def ReLU(self):
        return max(self.x, 0)
    
    def Leaky_ReLU(self, alpha = 0.01):
        x = self.x
        if (x >= 0):
            return x
        else:
            return alpha * x
    
    def sigmoid(self):
        return 1 / (1 + np.exp(-self.x))
    
    def tanh(self):
        return np.tanh(self.x)
    
    def softmax(self):
        x = self.x

        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        sums = np.sum(exp_vals, axis=1, keepdims=True)

        return exp_vals / sums

    
x = np.array([[1.0, 2.0, 3.0],
              [3.0, 0.5, -1.0]])
    
AF = activation_functions(x)


print(AF.softmax())        
    
