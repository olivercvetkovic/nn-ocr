import mnist_loader
from nn import NeuralNetwork, INITIALISING

training_data: list
validation_data: list
test_data: list
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

NN: NeuralNetwork = INITIALISING([784, 16, 16,  10])
print("finished initialising -> beginning training")

NN.SGD(training_data=training_data, testing_data=test_data, learning_rate=3, epochs=30, mini_batch_size=10)

print("finished training")
input("Press Enter to continue...")

# while True:

#     draw_number()

#     input = image_to_vector()

#     NN.compute(input=input)

#     user_input = input("type carry")

