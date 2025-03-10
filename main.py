import mnist_loader
import numpy as np
from nn import NeuralNetwork, INITIALISING
from gui import draw_number, vectorise_image

training_data: list
validation_data: list
test_data: list
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

NN: NeuralNetwork = INITIALISING([784, 30,  10])
print("finished initialising -> beginning training")

NN.SGD(training_data=training_data, testing_data=test_data, learning_rate=3, epochs=30, mini_batch_size=10)

print("finished training")
input("Press Enter to continue...")

while True:

    # Draw number
    draw_number(file_path="number.png")

    # Convert to vector to input to NN
    x: np.ndarray = vectorise_image(image_file="number.png")

    # Output of NN
    output: tuple[int, float] = NN.compute(input=x)
    print("The number on the image is:", output[0], "with probability", output[1])
