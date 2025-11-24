# Single Neuron
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = 0
for a, w in zip(inputs, weights):
    output += a * w
    
output += bias

print("Single Neuron output : ", output)

# Multiple Neuron
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
