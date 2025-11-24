inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = 0
for a, w in zip(inputs, weights):
    output += a * w
    
output += bias

print(output)