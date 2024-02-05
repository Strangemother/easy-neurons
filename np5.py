import numpy as np

# Correcting the dimensionality issues for the updated network structure

# Encoding letters to numbers
letter_to_num = {'w': 0.1, 'h': 0.2, 'e': 0.3, 'n': 0.4, 'a': 0.5, 't': 0.6}
num_to_letter = {v: k for k, v in letter_to_num.items()}


# Training data: (input, expected output)
training_data = [(0.1, 0.2),
                 (0.1, 0.2), # add twice fixed `w -> h`
                 (0.2, 0.3),
                 (0.3, 0.4),  # when
                 (0.2, 0.5),
                 (0.5, 0.6)  # what (adjusted based on your example)
                 ]
training_data_1_1 = training_data

# Reinitialize weights and biases for two hidden neurons, correcting initial mistake
np.random.seed(42)  # Resetting the seed for consistency

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize_network(input_size, hidden_layers, output_size):
    layers = [input_size] + hidden_layers + [output_size]
    weights = []
    biases = []
    print(f'initialize_network {layers=}')

    l = len(layers) - 1

    for i in range(l):
        weight = np.random.uniform(-1, 1, (layers[i], layers[i+1]))
        bias = np.random.uniform(-1, 1, (layers[i+1],))
        print(f'  #{i+1}/{l} {weight=}')
        print(f'         {bias=}\n')
        weights.append(weight)
        biases.append(bias)

    return weights, biases


def forward_pass(input_data, weights, biases):
    activations = [input_data]
    for i in range(len(weights)):
        # on error: likely due to the input shape not matching the input nodes.
        input_data = np.dot(input_data, weights[i]) + biases[i]
        input_data = sigmoid(input_data)
        activations.append(input_data)
    return activations

make_activations = forward_pass

def forward_pass_one(input_data, weights, biases):
    # # Generalized forward pass through all layers
    # for i in range(len(weights)):
    #     input_data = np.dot(input_data, weights[i]) + biases[i]
    #     input_data = sigmoid(input_data)
    # return input_data
    return forward_pass(input_data, weights, biases)[-1]

import numpy as np


def train_network(data, weights, biases, learning_rate=0.1, epochs=10000):
    for epoch in range(epochs):
        total_error = 0
        for input_val, expected_output in data:
            total_error += train_once(input_val, expected_output, learning_rate, weights, biases)
            # total_error += train_once_with_loss(input_val, ((expected_output,), False), learning_rate, weights, biases)
        # print out the error every so often
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Total Error: {total_error}")

    return weights, biases


def train_once(input_val, expected_output, learning_rate, weights, biases):
    # Forward pass: Store activations for each layer
    activations = forward_pass(input_val, weights, biases)

    # Backward pass
    # Calculate output error
    output_error = expected_output - activations[-1]
    total_error = np.sum(output_error ** 2)

    # Calculate gradient for output layer
    d_error = output_error * sigmoid_derivative(activations[-1])

    for i in reversed(range(len(weights))):
        # Calculate error for the current layer
        d_weights = np.outer(activations[i], d_error)
        d_biases = d_error

        # Update weights and biases
        weights[i] += learning_rate * d_weights
        # cannot be: `biases[i] += learning_rate * d_biases` due to numpy
        biases[i] = biases[i] + (learning_rate * d_biases)

        # Propagate the error backward
        if i > 0:
            d_error = np.dot(d_error, weights[i].T) * sigmoid_derivative(activations[i])
    return total_error


def custom_loss(predicted_output, true_output, is_negative_example):
    if is_negative_example:
        # Assuming the true_output for negative examples is what we want to avoid
        # This penalizes the model for predicting close to the true_output
        return (1 - abs(true_output - predicted_output)) ** 2
    else:
        # Standard MSE loss for positive examples
        return (true_output - predicted_output) ** 2

def train_once_with_loss(input_val, expected_output_info, learning_rate, weights, biases):
    # Adjust input to handle the new expected_output format
    expected_output, is_negative_example = expected_output_info

    # Forward pass: Store activations for each layer
    activations = forward_pass(input_val, weights, biases)

    # Backward pass
    # Calculate output error using custom loss function
    output_error = np.array([
        custom_loss(activation, expected_output[i], is_negative_example)
        for i, activation in enumerate(activations[-1])
    ])
    total_error = np.sum(output_error)

    # Calculate gradient for output layer
    d_error = output_error * sigmoid_derivative(activations[-1])

    for i in reversed(range(len(weights))):
        # Calculate error for the current layer
        d_weights = np.outer(activations[i], d_error)
        d_biases = d_error

        # Update weights and biases
        weights[i] += learning_rate * d_weights
        biases[i] = biases[i] + (learning_rate * d_biases)

        # Propagate the error backward if not in the first layer
        if i > 0:
            d_error = np.dot(d_error, weights[i].T) * sigmoid_derivative(activations[i])

    return total_error


def convert_to_numpy(val):
    return np.array([val]) if not isinstance(val, np.ndarray) else val



def predict(input_val, weights, biases, letter_to_num, num_to_letter):
    # Ensure input is in the correct format (numpy array)
    input_val = convert_to_numpy(input_val)

    v = forward_pass_one(input_val, weights, biases)
    # Generalized forward pass through all layers
    for i in range(len(weights)):
        input_val = np.dot(input_val, weights[i]) + biases[i]
        input_val = sigmoid(input_val)
    assert v == input_val
    # Convert the final output to the closest letter
    # Here, we're assuming the output is a single number. Adjust as needed for multiple outputs.
    final_output = v[0]  # Assuming single output for simplicity
    closest_num = min(letter_to_num.values(), key=lambda x: abs(x - final_output))
    predicted_letter = num_to_letter[closest_num]

    return predicted_letter


def predict_with_details(input_val, weights, biases, threshold=None):
    # Ensure input is in the correct format (numpy array)
    input_val = np.array([input_val]) if not isinstance(input_val, np.ndarray) else input_val

    # Generalized forward pass through all layers
    for i in range(len(weights)):
        input_val = np.dot(input_val, weights[i]) + biases[i]
        input_val = sigmoid(input_val)

    # If a threshold is specified, filter results
    if threshold is not None:
        filtered_results = [(i, activation) for i, activation in enumerate(input_val[0]) if activation > threshold]
        # Sort by activation level, highest first
        sorted_filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
        return sorted_filtered_results

    # Return all results sorted by activation level, highest first
    all_results = [(i, activation) for i, activation in enumerate(input_val[0])]
    sorted_all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    return sorted_all_results

# Example usage
# Assuming 'weights' and 'biases' are your trained network parameters
def predict_many(value, threshold=0.5):  # Example threshold, adjust as needed
    prediction_results = predict_with_details(value, weights, biases, threshold=threshold)

    # For converting indices back to letters or labels, you'd map each index in the results to its corresponding label
    for index, activation in prediction_results:
        print(f"Label #{index} (or corresponding letter), Activation: {activation}")

    return prediction_results


def predict_next_letter_updated(char):
    return predict(letter_to_num[char],
        weights,
        biases,
        letter_to_num,
        num_to_letter)  # Predict next letter from 'a'


# Example usage
# input_size = 2
# hidden_layers = [2, 4, 2]
# output_size = 2

input_size = 1
hidden_layers = [3,3]
output_size = 1

weights, biases = initialize_network(input_size, hidden_layers, output_size)

# Example usage
# weights, biases = initialize_network(1, [2], 1)
array = np.array

x_weights, x_biases = (
                    [
                        array([[ 2.32254687, -3.86480048]]),
                        array([[ 2.04736964], [-3.87521947]])
                    ],
                    [
                        array([[-0.36708326,  0.07149161]]),
                        array([[-0.45136009]])
                    ]
                )

## more fitting
x_weights, x_biases = (
        [
            array([[ 2.63878401, -7.06131613]]),
            array([[ 1.92740867], [-5.65455804]])
        ],
        [
            array([[ 0.0479399 , -0.52078492]]),
            array([[-1.09919102]])
        ]
    )

x_weights, x_biases = (
        [
            array([[4.45372861, 0.39046513]]),
            array([[ 4.56941577], [-0.37568509]])
        ],
        [
            array([[-0.31594015,  0.06035938]]),
            array([[-3.37011245]])
        ]
    )
"""
Overfitting:


>>> weights
[array([[ 2.42858316, -4.46958476]]), array([[ 1.97726357],
       [-4.24062114]])]

>>> do_all()
Predicted letter: w -> h, "True"
Predicted letter: h -> e, "True"
Predicted letter: e -> n, "True"
Predicted letter: a -> t, "True"

>>> train_network(training_data_1_1, weights, biases, epochs=10000)
Epoch 0, Total Error: 0.03185295994603337
Epoch 1000, Total Error: 0.03175548113234771
Epoch 2000, Total Error: 0.03166216527864295
Epoch 3000, Total Error: 0.03157277662424882
Epoch 4000, Total Error: 0.031487099791611524
Epoch 5000, Total Error: 0.03140493727094395
Epoch 6000, Total Error: 0.03132610725084045
Epoch 7000, Total Error: 0.03125044175043587
Epoch 8000, Total Error: 0.031177785012518236
Epoch 9000, Total Error: 0.03110799212134287
([array([[ 2.50577094, -4.99144597]]), array([[ 1.9270364 ],
       [-4.53543286]])], [array([[-0.10403118, -0.28071243]]), array([[-0.79685587]])])

>>> do_all()
Predicted letter: w -> h, "True"
Predicted letter: h -> e, "True"
Predicted letter: e -> a, "False"
Predicted letter: a -> t, "True"

"""


# Initialize the network with 1 input node, a hidden layer with 2 nodes, and 1 output node
# weights, biases = initialize_network(1, [2], 1)

# Add your training data and training loop as per the logic above
#
#
# Here you would add your training loop, including forward and backward passes
# Since this is a conceptual example, the training loop, data preparation, and detailed implementation of backpropagation are omitted
train_network(training_data_1_1, weights, biases, epochs=1000)

# After training, you could use the forward_pass function to make predictions
# Example input for prediction (make sure your input is shaped correctly)
input_val = np.array([0.1])  # Example input
activations = forward_pass(input_val, weights, biases)

# The final activation is the prediction
prediction = activations[-1]
print(f"Prediction: {prediction}")


def do_one(char, l):
    v = predict_next_letter_updated(char)
    print(f'Predicted letter: {char} -> {v}, "{v == l}"')
    return v

def do_all():
    v = do_one('w', 'h')
    v = do_one('h', 'e')
    v = do_one('e', 'n')
    v = do_one('a', 't')

do_all()

predict_many(0.1)

# ===


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.weights_input_to_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_to_hidden = np.random.uniform(-1, 1, (hidden_size, hidden_size))
        self.weights_hidden_to_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.hidden_state = np.zeros((hidden_size,))

    def forward(self, input_val):
        # Combine input with previous hidden state
        self.hidden_state = sigmoid(np.dot(input_val, self.weights_input_to_hidden) +
                                    np.dot(self.hidden_state, self.weights_hidden_to_hidden))
        output = sigmoid(np.dot(self.hidden_state, self.weights_hidden_to_output))
        return output

    def predict(self, input_sequence):
        outputs = []
        for input_val in input_sequence:
            output = self.forward(input_val)
            outputs.append(output)
        return outputs

# Example usage
rnn = SimpleRNN(input_size=1, hidden_size=10, output_size=1)  # Example sizes

# Example input sequence (e.g., time steps)
input_sequence = [np.array([x]) for x in np.linspace(0, 1, 10)]  # Generating example inputs

# Predicting the output sequence
# output_sequence = rnn.predict(input_sequence)
# Printing the output sequence
# print(f'{output_sequence=}')


# ---


# Assuming a simple RNN structure
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.U = np.random.uniform(-1, 1, (input_size, hidden_size))  # Weights for input
        self.W = np.random.uniform(-1, 1, (hidden_size, hidden_size)) # Weights for hidden-to-hidden
        self.V = np.random.uniform(-1, 1, (hidden_size, output_size)) # Weights for hidden-to-output

        self.bh = np.zeros((hidden_size,))  # Biases for hidden layer
        self.bo = np.zeros((output_size,))  # Biases for output layer

    def forward(self, inputs):
        # Initialize hidden state and outputs
        h = np.zeros((self.hidden_size,))
        outputs = []

        for x in inputs:
            h = sigmoid(np.dot(x, self.U) + np.dot(h, self.W) + self.bh)
            y = sigmoid(np.dot(h, self.V) + self.bo)
            outputs.append(y)

        return outputs, h

    def train(self, inputs, targets, learning_rate=0.1):
        # Forward pass
        outputs, h = self.forward(inputs)

        # Initialize gradients
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        dbh, dbo = np.zeros_like(self.bh), np.zeros_like(self.bo)

        # Error at output
        error = outputs[-1] - targets[-1]

        # Gradient of error with respect to output weights V and biases bo
        dV += np.dot(h.reshape(-1, 1), error.reshape(1, -1))
        dbo += error

        # Backpropagate through time (for one step here, extend as needed)
        dh = np.dot(self.V, error) * sigmoid_derivative(h)

        # Update for U and W considering the single back step
        # This would be in a loop for multiple steps, accumulating gradients
        dU += np.dot(inputs[-1].reshape(-1, 1), dh.reshape(1, -1))
        dW += np.dot(h.reshape(-1, 1), dh.reshape(1, -1))
        dbh += dh

        # Parameter update
        self.U -= learning_rate * dU
        self.W -= learning_rate * dW
        self.V -= learning_rate * dV
        self.bh -= learning_rate * dbh
        self.bo -= learning_rate * dbo

    def predict(self, input_sequence):
        """Generate a prediction for a given input sequence."""
        # Convert input sequence to numpy array, if not already
        if not isinstance(input_sequence, np.ndarray):
            input_sequence = np.array(input_sequence)

        # Reset hidden state for prediction
        self.hidden_state = np.zeros((self.hidden_size,))

        # Run the forward pass
        predictions = self.forward(input_sequence)

        # Optionally, you might want to convert the raw output (sigmoid activations) to a more interpretable form
        # For simplicity, this example returns the raw predictions
        return predictions

# Example usage
rnn = SimpleRNN(input_size=2, hidden_size=5, output_size=1)
inputs = [np.array([0.5, -0.2]), np.array([0.1, 0.4])]  # Example input sequence
targets = [np.array([0.6]), np.array([0.9])]  # Example target sequence
# rnn.train(inputs, targets, learning_rate=0.1)

# Generate some sample input data
# For instance, two timesteps with input size of 2
input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]])

# Get predictions
predictions = rnn.predict(input_sequence)
# print(f'{predictions=}')