import numpy as np

# initialize inputs (all stacked)
x = np.eye(8)  # inputs a.k.a. first layer


# initialize weights
theta1 = np.random.random(size=(8, 3)) - 0.5
theta2 = np.random.random(size=(3, 8)) - 0.5
thetas = [theta1, theta2]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x, thetas):
    """Feedforward pass including all intermediary stepss"""
    outputs = []
    for t in thetas:
        outputs.append(x)
        x = np.dot(x, t)
    activations = list(map(sigmoid, outputs))
    return outputs, activations

# layer1: sigmoid(Xb1)
# layer2: sigmoid((sigmoid(Xb1)b2)
# error e: (1 / 2) * (layer2 - y) ^ 2

# de/db2 = de/dlayer2 .* dlayer2/a1 .* a1/dz1 .* dz1/db2
# de/db2 = (layer2 - y) .* (a1 * (1 - a1)) .* (a1 * z1) .* (z1)

# de/db1 = de/dlayer2

def backpropagate(x, y, thetas):
    error = x - y
    delta = None
    for t in thetas:
        # delta = 
        pass
    pass



