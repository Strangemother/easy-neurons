import numpy as np
# Correcting the dimensionality issues for the updated network structure

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Encoding letters to numbers
letter_to_num = {'w': 0.1, 'h': 0.2, 'e': 0.3, 'n': 0.4, 'a': 0.5, 't': 0.6}
num_to_letter = {v: k for k, v in letter_to_num.items()}

# Training data: (input, expected output)
training_data = [(0.1, 0.2),
                 (0.2, 0.3),
                 (0.3, 0.4),  # when
                 (0.2, 0.5),
                 (0.5, 0.6)  # what (adjusted based on your example)
                 ]


# Reinitialize weights and biases for two hidden neurons, correcting initial mistake
np.random.seed(42)  # Resetting the seed for consistency
# Assume sigmoid and its derivative remain the same

def new_cache(input_size=2, output_size=2, hidden_size=2):
    # Adjust for 2 inputs and 2 outputs
    return {
        'weights_input_to_hidden': np.random.uniform(-1, 1, (input_size, hidden_size)),
        'weights_hidden_to_output': np.random.uniform(-1, 1, (hidden_size, output_size)),
        'bias_hidden': np.random.uniform(-1, 1, (hidden_size,)),
        'bias_output': np.random.uniform(-1, 1, (output_size,)),
    }

# Example updated to 2 inputs and 2 outputs
training_data = [
    # Imagine these numbers represent encoded pairs of letters
    (np.array([0.1, 0.2]), np.array([0.3, 0.4])),  # Encoded equivalent of "ab" -> "cd"
    # Add more training examples as needed
]

def train(data, cache, learning_rate=0.1, epochs=10000):
    for epoch in range(epochs):
        for input_val, expected_output in data:
            # Forward pass
            hidden_layer_input = np.dot(input_val, cache['weights_input_to_hidden']) + cache['bias_hidden']
            hidden_layer_output = sigmoid(hidden_layer_input)
            final_output_input = np.dot(hidden_layer_output, cache['weights_hidden_to_output']) + cache['bias_output']
            final_output = sigmoid(final_output_input)

            # Backward pass
            output_error = expected_output - final_output
            d_final_output = output_error * sigmoid_derivative(final_output)
            hidden_layer_error = d_final_output.dot(cache['weights_hidden_to_output'].T)
            d_hidden_layer = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            cache['weights_hidden_to_output'] += np.outer(hidden_layer_output, d_final_output) * learning_rate
            cache['bias_output'] += d_final_output * learning_rate
            cache['weights_input_to_hidden'] += np.outer(input_val, d_hidden_layer) * learning_rate
            cache['bias_hidden'] += d_hidden_layer * learning_rate

# Prediction function needs to accept and return pairs of numbers
def predict(input_val, cache):
    hidden_layer_input = np.dot(input_val, cache['weights_input_to_hidden']) + cache['bias_hidden']
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output_input = np.dot(hidden_layer_output, cache['weights_hidden_to_output']) + cache['bias_output']
    final_output = sigmoid(final_output_input)
    return final_output

# Initialize network
cache = new_cache()

# Train network
train(training_data, cache)

# Make a prediction
input_example = np.array([0.1, 0.2])  # Example input
prediction = predict(input_example, cache)
print(f'Prediction for {input_example}: {prediction}')
