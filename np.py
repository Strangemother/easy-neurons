import numpy as np

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

# Initialize weights and biases
input_size = 1
output_size = 1
hidden_size = 1

np.random.seed(42)  # For reproducibility
weights_input_to_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
weights_hidden_to_output = np.random.uniform(-1, 1, (hidden_size, output_size))
bias_hidden = np.random.uniform(-1, 1, (hidden_size,))
bias_output = np.random.uniform(-1, 1, (output_size,))

# Training parameters
learning_rate = 0.1
epochs = 30_000

# Training loop
for epoch in range(epochs):
    for input_val, expected_output in training_data:
        # Forward pass
        input_val = np.array([input_val])
        expected_output = np.array([expected_output])

        hidden_layer_input = np.dot(input_val, weights_input_to_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        final_output_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
        final_output = sigmoid(final_output_input)

        # Backward pass (error and gradient calculation)
        output_error = expected_output - final_output
        d_final_output = output_error * sigmoid_derivative(final_output)

        hidden_layer_error = d_final_output.dot(weights_hidden_to_output.T)
        d_hidden_layer = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_to_output += hidden_layer_output.T.dot(d_final_output) * learning_rate
        bias_output += np.sum(d_final_output, axis=0) * learning_rate

        weights_input_to_hidden += input_val.T.dot(d_hidden_layer) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0) * learning_rate

# Display final weights and biases for curiosity
print(weights_input_to_hidden, bias_hidden, weights_hidden_to_output, bias_output)

# Prediction function
def predict_next_letter(input_val):
    # Convert input letter to numerical form
    input_val = np.array([input_val])

    # Forward pass
    hidden_layer_input = np.dot(input_val, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_output_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    final_output = sigmoid(final_output_input)

    # Convert numerical output to closest letter encoding
    closest_num = min(letter_to_num.values(), key=lambda x: abs(x - final_output))
    predicted_letter = num_to_letter[closest_num]

    return predicted_letter

# Predict the next letter after 'h'
predicted_letter_after_h = predict_next_letter(letter_to_num['h'])


print(f'{predicted_letter_after_h=}')
