import numpy as np

# Single Neuron
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = 0
for a, w in zip(inputs, weights):
    output += a * w
    
output += bias

print("Single Neuron output : ", output)

# Multiple Neurons
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

weights = [weights1, weights2, weights3]

bias1 = 2
bias2 = 3
bias3 = 0.5

biases = [bias1, bias2, bias3]

outputs = []

for b, weight in zip(biases, weights):
    output = 0
    for a, w in zip(inputs, weight):
        output += a * w
    
    output += b
    
    outputs.append(output)
    
print("Mulutiple Neuron output : ", outputs)

# One layer
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] # (3, 4)

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]] # (3, 4)
 
biases = [2.0, 3.0, 0.5] 

outputs = np.dot(inputs, np.array(weights).T) + biases

print("Layer output : \n", outputs)

# Multiple layers
# Layer 1
inputs1 = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] # (3, 4)

weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]] # (3, 4)
 
biases1 = [2.0, 3.0, 0.5] 

# Layer2
weights2 = [[0.1, -0.14, 0.5],
          [-0.5, 0.12, -0.33],
          [-0.44, 0.73, -0.13]] # (3, 3)
 
biases2 = [-1.0, 2.0, -0.5]

layer1_output = np.dot(inputs1, np.asarray(weights1).T) + biases1  # (3, 3)
layer2_output = np.dot(layer1_output, np.asarray(weights2).T) + biases2

print("Layers output : \n", layer2_output)

