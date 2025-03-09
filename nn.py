from turtle import forward
import numpy as np
from collections.abc import Callable
import random


# Defining the Neuron and its methods
class Neuron:
    def __init__(self, activation_function: Callable[[float], float], bias: float, w: np.ndarray) -> None:
        self.activation_function = activation_function
        self.bias = bias
        self.w = w

    def __str__(self):
        return f"bias: {self.bias}\nweight vector: {self.w}\n"

    def bias_str(self):
        return f"{self.bias}"

    def w_str(self):
        return f"{self.w}"

    def pre_activation(self, input: np.ndarray) -> float:
        if input.size != self.w.size:
            raise Exception("Dimension of input vector and weight vector differ")
        else:
            return np.dot(input, self.w) + self.bias

    def activation(self, pre_activation: float) -> float:
        return self.activation_function(pre_activation)


# Defining a Layer and its methods
class Layer:
    def __init__(self, neurons: list[Neuron]) -> None:
        self.neurons = neurons
        self.W = np.stack([neuron.w for neuron in neurons])
        self.b = np.array([neuron.bias for neuron in neurons]).reshape(-1, 1)
        self.activation_functions = [neuron.activation_function for neuron in neurons]

    def __str__(self):
        return f"weight matriinput:\n{self.W} \n \nbias_vector:\n{self.b} \n"

    def b_str(self):
        return f"{self.b}"

    def W_str(self):
        return f"{self.W}"

    def pre_activations(self, input: np.ndarray) -> np.ndarray:
        return np.matmul(self.W, input) + self.b

    def activations(self, pre_activations: np.ndarray) -> np.ndarray:
        return np.array([activation_function(pre_activation) for activation_function, pre_activation in zip(self.activation_functions, pre_activations)])


# Defining a network and its methods
class NeuralNetwork:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        self.depth = len(layers)

    def compute(self, input: np.ndarray) -> int:
        pre_activations: list[np.ndarray]
        activations: list[np.ndarray]
        
        pre_activations, activations = self.forward_pass(input=input)
        output = np.argmax(activations[-1])
    
        return int(output)

    def forward_pass(self, input: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        a: np.ndarray = input

        pre_activations: list[np.ndarray] = []
        activations: list[np.ndarray] = [a]

        for l in range(self.depth):
            layer: Layer = self.layers[l]

            z = layer.pre_activations(input=a)
            pre_activations.append(z)

            a = layer.activations(pre_activations=z)
            activations.append(a)

        return (pre_activations, activations)

    def depth_str(self):
        return f"{self.depth}"

    def cost_derivative(self, activations: list, y: np.ndarray) -> np.ndarray:
        return activations - y

    def backward_pass(self, pre_activations: list, activations: list, y: np.ndarray) -> list[np.ndarray]:
        sensitivities: list[np.ndarray] = []

        # for l = L
        sensitivities_l: np.ndarray = sigmoid_prime(pre_activations[-1]) * self.cost_derivative(activations=activations[-1], y=y)
        sensitivities.append(sensitivities_l)

        # for l = L-1,...1
        for l in range(1, self.depth):
            layer: Layer = self.layers[-l]

            sensitivities_l: np.ndarray = sigmoid_prime(pre_activations[-l-1]) * np.matmul(layer.W.T, sensitivities[-l])
            sensitivities.insert(0, sensitivities_l)

        return sensitivities

    def backpropagation(self, input: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:

        # Forward pass algorithm
        pre_activations: list[np.ndarray]
        activations: list[np.ndarray]
        pre_activations, activations = self.forward_pass(input=input)

        # Backward pass algorithm
        sensitivities: list[np.ndarray] = self.backward_pass(pre_activations=pre_activations, activations=activations, y=y)

        nablaC_b: list[np.ndarray] = sensitivities

        nablaC_W: list[np.ndarray] = []
        for l in range(self.depth):
            s: np.ndarray = sensitivities[l]
            a: np.ndarray = activations[l]

            nablaC_W_l: np.ndarray = np.outer(s, a)
            nablaC_W.append(nablaC_W_l)

        return (nablaC_W, nablaC_b)

    def update_parameters(self, data: list, learning_rate: float) -> None:
        nabla_W_sum = [np.zeros_like(layer.W) for layer in self.layers]
        nabla_b_sum = [np.zeros_like(layer.b) for layer in self.layers]

        for x, y in data:
            nablaC_W, nablaC_b = self.backpropagation(input=x, y=y)

            for l in range(self.depth):
                nabla_W_sum[l] += nablaC_W[l]
                nabla_b_sum[l] += nablaC_b[l]

        batch_size = len(data)
        for l in range(self.depth):
            self.layers[l].W -= (learning_rate / batch_size) * nabla_W_sum[l]
            self.layers[l].b -= (learning_rate / batch_size) * nabla_b_sum[l]

    def SGD(self, training_data: list, testing_data: list, epochs: int, mini_batch_size: int, learning_rate: float) -> None:

        for j in range(epochs):
            random.shuffle(training_data)
            data_sets = [training_data[k: k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]

            for data in data_sets:
                self.update_parameters(data=data, learning_rate=learning_rate)

            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(testing_data=testing_data), len(testing_data)))

    def evaluate(self, testing_data: list) -> float:
        training_results: list = []

        for x, y in testing_data:
            activations: list[np.ndarray]
            outputs = self.forward_pass(x)
            activations = outputs[-1]

            training_results.append((np.argmax(activations[-1]), y))

        return sum(int(x == y) for (x, y) in training_results)
    


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z=z)*(1-sigmoid(z))


def INITIALISING(sizes: list) -> NeuralNetwork:

    layers: list[Layer] = []
    for input_size, output_size in zip(sizes[:-1], sizes[1:]):
        neurons: list = []

        for i in range(output_size):
            neuron = Neuron(activation_function=sigmoid, bias=np.random.randn(), w=np.random.randn(input_size))
            neurons.append(neuron)

        layer: Layer = Layer(neurons=neurons)
        layers.append(layer)

    return NeuralNetwork(layers=layers)


def ReLU(x):
    return np.maximum(np.zeros_like(x), x)


def ReLU_prime(x):

    return np.where(x < 0, 0, 1)


# # Layer 1
# neuron_1_1: Neuron = Neuron(activation_function=sigmoid, bias=1, w=np.array([1, 2]))
# neuron_1_2: Neuron = Neuron(activation_function=sigmoid, bias=0, w=np.array([0, 1]))

# layer_1: Layer = Layer([neuron_1_1, neuron_1_2])

# # Layer 2
# neuron_2_1: Neuron = Neuron(activation_function=sigmoid, bias=2, w=np.array([1, 3]))
# neuron_2_2: Neuron = Neuron(activation_function=sigmoid, bias=-1, w=np.array([2, 0]))
# neuron_2_3: Neuron = Neuron(activation_function=sigmoid, bias=0, w=np.array([3, 2]))

# layer_2: Layer = Layer([neuron_2_1, neuron_2_2, neuron_2_3])

# print(layer_1.W, "\n\n", layer_2.W)

# # network
# nn = NeuralNetwork([layer_1, layer_2])

# input = np.array([1, 1]).reshape(-1, 1)


# pre_activations, activations = nn.forward_pass(input=input)

# print("FORWARD PASS RESULTS: \n pre_activations \n", pre_activations, "\n activations \n", activations)

# print("\n TESITNG BACKWARD PASS \n")
# sensitivities = nn.backward_pass(pre_activations=pre_activations, activations=activations, y=np.array([0, 1, 0]).reshape(-1, 1))
# print("BACKWARD PASS RESULTS (sensitivities): \n", sensitivities)

# print("\n TESTING BACKPROP \n")
# nablaC_W, nablaC_b = nn.backpropagation(input=input, y=np.array([0, 1, 0]).reshape(-1, 1))

# print("BACKPROP RESULTS: \n changes in W \n", nablaC_W, "\n changes in B \n", nablaC_b)