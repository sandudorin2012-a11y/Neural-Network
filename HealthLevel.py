import numpy as np

inputs = [7.5 , 8 , 4, 3 ,2.5]
weights = [0.2, 0.15, 0.1, -0.25, 0.1]
bias = -2
weighted_input = [inputs[i] * weights[i] for i in range(len(inputs))]
total_input = sum(weighted_input)+bias
print(total_input)
def sigmoid(x):
    return 1/ (1 + np.exp(-x))
activated_output = sigmoid(total_input)
print("Health Level:", activated_output)