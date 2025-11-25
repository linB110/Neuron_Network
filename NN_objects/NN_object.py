import numpy as np

np.random.seed(0)

class NN_layer:
    def __init__(self, n_inputs ,n_neurons):
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, np.asarray(self.weights).T) + self.biases
    
    def normalization(self, input, lower_bound = 0, upper_bound = 1):
        max_val = np.max(input)
        min_val = np.min(input)
        
        if max_val == min_val:
            return np.full_like(input, lower_bound)
        
        sacale_factor = (upper_bound - lower_bound) / (max_val - min_val)
        
        return lower_bound + (input - min_val) * sacale_factor
    
    def normalization(input, lower_bound = 0, upper_bound = 1):
            max_val = np.max(input)
            min_val = np.min(input)
            
            if max_val == min_val:
                return np.full_like(input, lower_bound)
            
            sacale_factor = (upper_bound - lower_bound) / (max_val - min_val)
            
            return lower_bound + (input - min_val) * sacale_factor

# usage
input = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] 

layer1 = NN_layer(4, 5)
layer2 = NN_layer(5, 2)

layer1.forward(input)
print("Layer 1 : \n", layer1.output)
layer2.forward(layer1.output)
print("Layer 2 : \n", layer2.output)
