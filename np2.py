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

def new_cache(input_size=1, output_size=1, hidden_size=2):
    # Initialize weights and biases
    return {
        'weights_input_to_hidden': np.random.uniform(-1, 1, (input_size, hidden_size)),  # Adjusted for two hidden neurons
        'weights_hidden_to_output': np.random.uniform(-1, 1, (hidden_size, output_size)),  # Adjusted for two hidden neurons output
        'bias_hidden': np.random.uniform(-1, 1, (hidden_size,)),  # Bias for two hidden neurons
        'bias_output': np.random.uniform(-1, 1, (output_size,)),
    }


cache = new_cache()

# Training parameters
def train(data, learning_rate=0.1, epochs=10000):
    # Corrected training loop with dimensionality adjustments
    for epoch in range(epochs):
        for input_val, expected_output in data:
            # Forward pass
            input_val = np.array([input_val])
            expected_output = np.array([expected_output])

            hidden_layer_input = np.dot(input_val, cache['weights_input_to_hidden']) + cache['bias_hidden']
            hidden_layer_output = sigmoid(hidden_layer_input)

            final_output_input = np.dot(hidden_layer_output, cache['weights_hidden_to_output']) + cache['bias_output']
            final_output = sigmoid(final_output_input)

            # Backward pass
            output_error = expected_output - final_output
            d_final_output = output_error * sigmoid_derivative(final_output)

            hidden_layer_error = d_final_output.dot(cache['weights_hidden_to_output'].T)
            d_hidden_layer = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

            # Update weights and biases, ensuring correct shapes
            cache['weights_hidden_to_output'] += np.outer(hidden_layer_output, d_final_output) * learning_rate
            cache['bias_output'] += d_final_output * learning_rate

            cache['weights_input_to_hidden'] += np.outer(input_val, d_hidden_layer) * learning_rate
            cache['bias_hidden'] += d_hidden_layer * learning_rate

    # Displaying the corrected weights and biases after training
    print(cache['weights_input_to_hidden'],
            cache['bias_hidden'],
            cache['weights_hidden_to_output'],
            cache['bias_output'])

train(training_data)

# Updated prediction function for two hidden neurons
def predict_next_letter_updated(input_val):
    # Convert input letter to numerical form
    input_val = np.array([input_val])

    # Forward pass with updated network structure
    hidden_layer_input = np.dot(input_val, cache['weights_input_to_hidden']) + cache['bias_hidden']
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_output_input = np.dot(hidden_layer_output, cache['weights_hidden_to_output']) + cache['bias_output']
    final_output = sigmoid(final_output_input)

    # Convert numerical output to closest letter encoding
    closest_num = min(letter_to_num.values(), key=lambda x: abs(x - final_output))
    predicted_letter = num_to_letter[closest_num]

    return predicted_letter

# Let's predict the next letter after 'a' with the updated model
predicted_letter_after_a_updated = predict_next_letter_updated(letter_to_num['a'])
print(f'{predicted_letter_after_a_updated=}')

# Let's predict the next letter after 'e' with the updated model
predicted_letter_after_e_updated = predict_next_letter_updated(letter_to_num['e'])
print(f'{predicted_letter_after_e_updated=}')

