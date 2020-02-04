import numpy as np



class CauseNet:

    def __init__(self, input_dims, output_dims, num_hidden, hidden_dims,
    eta=0.15):
        assert num_hidden > 0
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_hidden = num_hidden
        self.hidden_dims = hidden_dims
        self.weight_matrices = []
        self.eta = eta

    def create_architecture(self):
        self.weight_matrices.append(
            (np.random.rand(self.hidden_dims, self.input_dims+1)-0.5)/1)
        for i in range(self.num_hidden-1):
            self.weight_matrices.append(
                (np.random.rand(self.hidden_dims, self.hidden_dims+1)-0.5)/1)
        self.weight_matrices.append(
            (np.random.rand(self.output_dims, self.hidden_dims+1)-0.5)/1)

    def show(self):
        for i in self.weight_matrices:
            print(f"{i}\n{i.shape}\n")

    def predict(self, x):
        batch_size = x.shape[1]
        x = np.concatenate((np.ones((1, batch_size)), x), axis=0)
        activations = [x]
        for theta in self.weight_matrices:
            z = np.dot(theta, x).reshape(-1, max(1, batch_size))
            z = self.sigmoid(z, deriv=False)
            x = np.concatenate((np.ones((1, batch_size)), z), axis=0)
            activations.append(x)
            # current_shape = x.shape
            # print(f"Output shape {current_shape}, moving on to next layer")
        activations[-1] = activations[-1][1:, :]
        return activations

    def backpropagate(self, x, y):
        batch_size = x.shape[1]
        activations = self.predict(x)
        error_deriv = np.subtract(activations[-1], y)
        delta = []
        for i in range(1, len(self.weight_matrices)+1):
            # print(f"step {i} of backpropagation")
            if len(delta) == 0:
                delta = np.multiply(
                    error_deriv,
                    self.sigmoid(activations[-i], deriv=True)
                )
            else:
                delta = np.multiply(
                    np.dot(
                        np.transpose(self.weight_matrices[-(i-1)]),
                        delta
                    ),
                    self.sigmoid(activations[-i], deriv=True)
                )[1:, :]
            # delta_shape = delta.shape
            # print(f"delta shape: {delta_shape}")
            gradient = np.dot(
                delta,
                np.transpose(activations[-(i+1)])
            )
            # grad_shape = gradient.shape
            # print(f"gradient shape: {grad_shape}")
            self.weight_matrices[-i] -= self.eta * (1/batch_size) * gradient
        

    def sigmoid(self, x, deriv=False):
        sig = 1/(1 + np.exp(-x))
        if deriv:
            return np.multiply(sig, 1-sig)
        else:
            return sig


if __name__ == "__main__":

    # TODO: we have no bias yet - why does this seem to be critical -> to have smth to subtract??

    cn = CauseNet(input_dims=8, output_dims=8, num_hidden=1, hidden_dims=3)
    cn.create_architecture()
    cn.show()
    inp = np.eye(8)
    # inp = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
    #                 [0, 0, 0, 0, 0, 0, 1, 0],
    #                 [0, 0, 0, 0, 0, 1, 0, 0],
    #                 [0, 0, 0, 0, 1, 0, 0, 0],
    #                 [0, 0, 0, 1, 0, 0, 0, 0]]).T

    # initial prediction
    result = cn.predict(inp)
    for i in result:
        print(np.round(i, 3))

    # train
    for i in range(10000):
        cn.backpropagate(inp, inp)

    # predict again
    result = cn.predict(inp)
    for i in result:
        print(np.round(i, 3))