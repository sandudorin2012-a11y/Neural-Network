import numpy as np

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

inputs = [19, 20]
bias = 15
weights = [0.7, 0.5]
weighted_inputs = [inputs[i] * weights[i] for i in range (len(inputs))]

total_input = sum(weighted_inputs) + bias

activated = sigmoid(total_input)
print(activated)