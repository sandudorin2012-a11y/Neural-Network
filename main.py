import numpy as np
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

inputs = [25, 70]
weights = [0.7, 0.3]
bias = -15

weighted_inputs = [inputs[i] * weights[i] for i in range(len(inputs))]
total_input = sum(weighted_inputs) + bias

activated_output = sigmoid(total_input)
print("The result of the neuron's work:", activated_output)