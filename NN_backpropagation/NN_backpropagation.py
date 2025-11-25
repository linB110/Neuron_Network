import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

N_samples = 200
Layers = [1, 10, 10, 10, 1]
Learning_rate = 0.1
Epoch = 10001

x_samples = np.random.uniform(low=0.0, high=2 * np.pi, size=(N_samples, 1))
y_samples = np.sin(x_samples) + np.random.normal(loc=0.0, scale=0.3, size=(N_samples, 1))

# plt.scatter(x_samples, y_samples)
# plt.show()

# activation functions
sigmoid = lambda x : 1 / (1 + np.exp(-x))
identity = lambda x : x
ReLU = lambda x: np.maximum(x, 0)

# derivative of activation functions
sigmoid_prime = lambda f : (1 - f) * f
identity_prime = lambda f : 1

# initialize weight, bias and activation functions
weight_matrices = []
bias_vectors = []
activation_functions = []
for input_layer, output_layer in zip(Layers[:-1], Layers[1:]):
    kernel_of_matrices_limit = np.sqrt(6 / (input_layer + output_layer))
    
    w = np.random.uniform(
        low=-kernel_of_matrices_limit,
        high=kernel_of_matrices_limit,
        size=(input_layer, output_layer)
    )
    
    b = np.zeros(output_layer)
    
    weight_matrices.append(w)
    bias_vectors.append(b)
    activation_functions.append(sigmoid)

activation_functions[-1] = identity

def forward_pass(input, weight, bias, activation_function):
    x = input
    
    for w, b ,f in zip(weight, bias, activation_function):
        x = x @ w + b
        x = f(x)
    
    return x   

def loss_forward(y_pred, y_true):
    delta = y_pred - y_true
    
    return 0.5 * np.mean(delta**2)

def loss_backward(y_pred, y_true):
    delta = y_pred - y_true
    
    N = delta.size
    
    return delta / N

def network_forward_and_backward(x, y_ref, weights, biases, activations, activations_derivatives):
    a = x

    # Store the intermediate activated states for the backward pass
    layer_states = [a, ]

    for W, b, f in zip(weights, biases, activations):
        a = a @ W
        a = a + b
        a = f(a)
        layer_states.append(a)
    
    y = a

    loss = loss_forward(y, y_ref)

    current_cotangent = loss_backward(y, y_ref)

    weight_gradients = []
    bias_gradients = []

    for W, f_prime, a_current, a_prev in zip(
        reversed(weights),
        reversed(activations_derivatives),
        reversed(layer_states[1:]),
        reversed(layer_states[:-1]),
    ):
        activated_state_cotangent = current_cotangent
        plus_bias_state_cotangent = activated_state_cotangent * f_prime(a_current)

        bias_grad = np.sum(plus_bias_state_cotangent, axis=0)

        state_cotangent = plus_bias_state_cotangent

        prev_activated_state_cotangent = state_cotangent @ W.T
        
        weight_grad = a_prev.T @ state_cotangent
        
        bias_gradients.append(bias_grad)
        weight_gradients.append(weight_grad)
        
        current_cotangent = prev_activated_state_cotangent
    
    return loss, reversed(weight_gradients), reversed(bias_gradients)
        
activations_derivatives = [sigmoid_prime] * (len(Layers) - 2) + [identity_prime]

# Training
loss_history = []
for epoch in range(Epoch):
    loss, weight_grad, bias_grad = network_forward_and_backward(x_samples, y_samples, weight_matrices, bias_vectors, activation_functions, activations_derivatives)
    
    for w, w_grad, b ,b_grad in zip(weight_matrices, weight_grad, bias_vectors, bias_grad):
        w -= Learning_rate * w_grad
        b -= Learning_rate * b_grad
    
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, loss : {loss}")
        
    loss_history.append(loss)

        
# Visualization        
# plt.scatter(x_samples, y_samples)
# plt.scatter(x_samples, forward_pass(x_samples, weight_matrices, bias_vectors, activation_functions))

plt.plot(loss_history)
plt.yscale("log")

plt.show()

