import numpy as np
np.set_printoptions(suppress=True)


class CauseNet:

    def __init__(self, input_dims, output_dims, num_hidden, hidden_dims,
    eta=0.15, reg=0.005):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_hidden = num_hidden
        self.hidden_dims = hidden_dims
        self.weight_matrices = []
        self.eta = eta
        self.reg = reg
        self.create_architecture()


    def create_architecture(self):
        if self.num_hidden > 0:
            self.weight_matrices.append(
                (np.random.rand(self.hidden_dims, self.input_dims+1)-0.5)/1)
            for i in range(self.num_hidden-1):
                self.weight_matrices.append(
                    (np.random.rand(self.hidden_dims, self.hidden_dims+1)-0.5)/1)
            self.weight_matrices.append(
                (np.random.rand(self.output_dims, self.hidden_dims+1)-0.5)/1)
        else:
            print("Creating single layer NN")
            self.weight_matrices.append(
                (np.random.rand(self.output_dims, self.input_dims+1)-0.5)/1)


    def show(self):
        for i, l in enumerate(self.weight_matrices):
            print(f"Weights {i} with shape {l.shape}")
            print(f"{np.round(l, 2)}\n")

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

            # # L2 regularization
            # gradient += np.multiply(self.reg, (self.weight_matrices[-i]))

            # # Smth like L1 regularization
            gradient += np.multiply(
                np.sign(self.weight_matrices[-i]),
                np.where(self.weight_matrices[-i] != 0, self.reg, 0)
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


def train(cn, x, y):

    # initial prediction
    result = cn.predict(x)
    print("\nPredictions before training:")
    for i in result:
        print(np.round(i, 3))
        print("\n")

    # train
    for i in range(30000):
        cn.backpropagate(x, y)

    # predict again
    print("\nPredictions after training:")
    result = cn.predict(x)
    for i in result:
        print(np.round(i, 3))
        print("\n")

    mse = np.mean(np.square(np.subtract(result[-1], y)))
    print(f"\n\nMSE: {mse}")
    



if __name__ == "__main__":

    # TODO: override predictions with candidate causal intermediaries where possible!
    # TODO: implement pretraining on causal stages!
    cn = CauseNet(input_dims=8, output_dims=8, num_hidden=1, hidden_dims=3)
    cn.show()
    x = np.eye(8)
    # inp = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
    #                 [0, 0, 0, 0, 0, 0, 1, 0],
    #                 [0, 0, 0, 0, 0, 1, 0, 0],
    #                 [0, 0, 0, 0, 1, 0, 0, 0],
    #                 [0, 0, 0, 1, 0, 0, 0, 0]]).T
    y = x
    train(cn, x, y)

