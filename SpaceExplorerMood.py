import numpy as np

inputs = [10, 10, 3, 5, 7]
weights = [0.3, 0.1, 0.5, 0.6, 0.1]
bias = -1
weighted_input = [inputs[i] * weights[i] for i in range(len(inputs))]
total_input = sum(weighted_input)+bias

def tanh(x):
    return np.tanh(x)
activated_output = tanh(total_input)
print("mood level",activated_output)