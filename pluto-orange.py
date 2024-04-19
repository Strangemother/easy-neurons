import numpy as np
# Correcting the dimensionality issues for the updated network structure

# Sigmoid activation function and its derivative
def sigmoid(x):
    # https://en.wikipedia.org/wiki/Exponential_function
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Encoding letters to numbers
letter_to_num = {
                 ' ': 0.0,
                 'a': 0.1,
                 'e': 0.2,
                 'p': 0.3,
                 'g': 0.4,
                 'l': 0.5,
                 'n': 0.6,
                 'o': 0.7,
                 'u': 0.8,
                 'r': 0.9,
                 't': 1.0,
                }


num_to_letter = {v: k for k, v in letter_to_num.items()}


training_data = [

    (np.array([0.3, 0.5, 0.8, 1.0, 0.7, 0.0]),
     np.array([0.7, 0.9, 0.1, 0.6, 0.4, 0.2])),

    (
     np.array([0.7, 0.9, 0.1, 0.6, 0.4, 0.2]),
    np.array([0.3, 0.5, 0.8, 1.0, 0.7, 0.0]),
     ),
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


def train(data, cache, learning_rate=0.1, epochs=10):
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


def closest_match(value, value_in, value_out):
    closest_num = min(value_in.values(), key=lambda x: abs(x - value))
    predicted_letter = value_out[closest_num]
    return predicted_letter


def pred(ie):
    print('Prediction')
    # Make a prediction
    prediction = predict(ie, cache)
    print(f'Prediction for {ie}: {prediction}')
    r = ()
    for p in prediction:
        v = closest_match(p, letter_to_num, num_to_letter)
        r += (v,)

    print(r)
    return r


def tp(e=10):
    # Train network
    print(pred(input_example))
    train(training_data, cache, epochs=e)
    print(pred(input_example))

def convert(value):
    res = [letter_to_num[x] for x in value]
    # res = [0.3, 0.5, 0.8, 1.0, 0.7, 0.0]
    return np.array(res)

# input_example = np.array([0.05, 0.13, 0.2, 0.3, 0.4, 0.0])
# input_example = np.array([0.3, 0.5, 0.8, 1.0, 0.7, 0.0])
input_example = convert('orange')

# Initialize network
cache = new_cache(6, 6, 12)

# Run about 5 times to train.
tp()