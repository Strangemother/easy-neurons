import numpy as np
from np6 import (sigmoid, sigmoid_derivative,# initialize_network,
                  forward_pass, forward_pass_one, train_network,
                  convert_to_numpy)

make_activations = forward_pass
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

array = np.array

x_weights_v1, x_biases_v1 = (
    [
        array([[ 2.32254687, -3.86480048]]),
        array([[ 2.04736964], [-3.87521947]])
    ],
    [
        array([[-0.36708326,  0.07149161]]),
        array([[-0.45136009]])
    ]
)

x_weights_v2, x_biases_v2 = (
    [
        array([[4.45372861, 0.39046513]]),
        array([[ 4.56941577], [-0.37568509]])
    ],
    [
        array([[-0.31594015,  0.06035938]]),
        array([[-3.37011245]])
    ]
)



def random_matrix(*structure):
    """
    """
    # Produce a flat array of random floats,
    # np.random.uniform(-1, 1, (2, ))
    # array([-0.77305296,  0.84938724])
    #  np.random.uniform(-1, 1, (3,2)) ## Array of 3 rows with 2 float per row
    #  array([[-0.32194042, -0.30158085],
    #         [ 0.45191136,  0.79422052],
    #         [ 0.77417285,  0.55975109]])
    return np.random.uniform(-1, 1, structure)


def initialize_network(input_size, hidden_layers, output_size):
    layers = [input_size] + hidden_layers + [output_size]
    weights = []
    biases = []
    print(f'new initialize_network {layers=}')

    l = len(layers) - 1

    for i in range(l):
        bias = random_matrix(layers[i+1])
        weight = random_matrix(layers[i], layers[i+1])
        # print(f'  #{i+1}/{l} {weight=}')
        # print(f'         {bias=}\n')
        weights.append(weight)
        biases.append(bias)

    return weights, biases


def main():
    # Example usage
    # input_size = 2
    # hidden_layers = [2, 4, 2]
    # output_size = 2
    global v1, v2, nn, nn2
    print('\n\n\n -- start \n\n')
    input_size = 1
    hidden_layers = [2]
    output_size = 1

    # weights, biases = initialize_network(input_size, hidden_layers, output_size)
    # Example usage
    # weights, biases = initialize_network(1, [2], 1)
    v1 = NN(wb=WeightsBiases(x_weights_v1, x_biases_v1))
    v2 = NN(wb=WeightsBiases(x_weights_v2, x_biases_v2))

    nn = NN(Shape(1,[1],1))
    nn.wb=nn.init_network()
    nn.train(training_data)
    nn.predict_next_letter('w')

    # Shape(1, [1], 1)
    nn2 = NN(wb=WeightsBiases(
            weights=[array([[-4.64938605]]), array([[-4.39859751]])],
            biases=[array([[0.38187714]]), array([[0.96090313]])],
        ))

    print(nn2.predict_next_letter('w'))

    do_all(nn)

    print('\n-- nn2')
    do_all(nn2)


class Shape(object):
    """A "shape" represents the structure foe `input, [hidden, ...], output`
    of a NN - that's it.
    It's built into this classy form so it's easier to work with when providing
    to a NN class.

        shape = Shape(1, [2,4,4,2], 2)
        NN(shape)
    """
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

    def __repr__(self):
        s = self
        cn = s.__class__.__name__
        return f"<{cn}({s.input_size}, {s.hidden_layers}, {s.output_size})>"

    @classmethod
    def infer(self, wb, flat=False):
        """Given an array ordered as a list of layers, return the
         input, [hidden], output of the weights, used to create a new
         shape from the weights.

            >>> nn = NN(Shape(2,[1,3,4,5,3,2],2))
            >>> wb = nn.init_network()
            >>> Shape.infer(wb.weights)
            [2, [1, 3, 4, 5, 3, 2], 2]
            >>> Shape.infer(wb.weights, flat=True)
            [2, 1, 3, 4, 5, 3, 2, 2]

            >> wb = WeightsBiases.from_shape(Shape(1,[2,3,4,3,2],2))
            >>> Shape.infer(wb, True)
            [1, 2, 3, 4, 3, 2, 2]

         During usage this isn't required, as a NN uses a shape _or_ the WB
         and will run through the weights without needing a pre-loaded 'shape'
        """
        weights = wb.weights
        biases = wb.biases
        hidden_sizes = (len(x) for x in weights[1:])
        if flat:
            return [len(weights[0]), *hidden_sizes, len(biases[-1])]
        return [len(weights[0]), list(hidden_sizes), len(biases[-1])]


class WeightsBiases(object):
    # A unit to keep weights and biases.
    # Useful for easier setup and save/restore
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def shape(self):
        """return a shape object infered by the weights"""
        return Shape.infer(self)

    @classmethod
    def from_shape(cls, shape):
        weights_biases = initialize_network(shape.input_size,
                                            shape.hidden_layers,
                                            shape.output_size
                                           )
        return WeightsBiases(*weights_biases)

    @classmethod
    def from_nn(cls, nn):
        """
            WeightsBiases.from_nn(NN(Shape(2,[3,2,1,3,4],2))).shape()
        """
        if nn.shape is not None:
            return cls.from_shape(nn.shape)
        return cls.from_shape(Shape(*Shape.infer(nn.wb)))
        # return WeightsBiases(*weights_biases)



class NN(object):
    """An extremely simple setup of a neural net. with exposed weights and biases
    for conceptualising potential shapes and layer structures.
    """
    def __init__(self, shape=None, wb=None):
        self.shape = shape
        self.wb = wb

    def init_network(self, shape=None):
        """
            NN(Shape(1,[2,3,2],1)).init_network().shape()
            [1, [2, 3, 2], 1]
        """
        shape = shape or self.shape
        weights_biases = initialize_network(shape.input_size,
                                            shape.hidden_layers,
                                            shape.output_size
                                           )
        wb = WeightsBiases(*weights_biases)
        return wb

    def train(self, data, wb=None, weights=None, biases=None, learning_rate=.1, epochs=5_000):
        wb = wb or self.wb
        if wb is not None:
            weights = wb.weights
            biases = wb.biases
        return self.iter_train_network(data, weights, biases,
                                       learning_rate=learning_rate, epochs=epochs)

    def iter_train_network(self, data, weights, biases, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for input_val, expected_output in data:
                total_error += self.train_once(input_val,
                                               expected_output,
                                               learning_rate, weights, biases)
                # total_error += self.train_once_with_loss(input_val,
                #                                     ((expected_output,), False),
                #                                     learning_rate, weights, biases)
            # print out the error every so often
            if epoch % 1000 == 0:
                print(f"Iter Epoch {epoch}, Total Error: {total_error}")

        return weights, biases

    def train_once(self, input_val, expected_output, learning_rate, weights, biases):
        """
        # Example of adjusting training data to include negative example flag
        training_data = [
            # Positive examples: ((input_features), (target_output, False))
            ((input_features_for_example_1), (target_output_for_example_1, False)),
            # Negative examples: ((input_features), (target_output, True))
            ((input_features_for_negative_example_1), (target_output_for_negative_example_1, True)),
        ]

        """
        # Forward pass: Store activations for each layer
        activations = self.forward_pass(input_val, weights, biases)

        # Backward pass
        # Calculate output error
        output_error = expected_output - activations[-1]
        total_error = np.sum(output_error ** 2)

        # Calculate gradient for output layer
        d_error = output_error * sigmoid_derivative(activations[-1])

        for i in reversed(range(len(weights))):
            d_error = self.calculate_current_layer(activations, i, d_error, learning_rate, weights, biases)
        # # Calculate error for the current layer
            # d_weights = np.outer(activations[i], d_error)
            # d_biases = d_error

            # # Update weights and biases
            # weights[i] += learning_rate * d_weights
            # # cannot be: `biases[i] +=` due to numpy
            # biases[i] = biases[i] + (learning_rate * d_biases)

            # # Propagate the error backward
            # if i <= 0:
            #     continue
            # # if i > 0:
            # d_error = np.dot(d_error, weights[i].T) * sigmoid_derivative(activations[i])

        return total_error

    def calculate_current_layer(self, activations, i, d_error, learning_rate, weights, biases):
        # Calculate error for the current layer
        d_weights = np.outer(activations[i], d_error)
        d_biases = d_error

        # Update weights and biases
        weights[i] += learning_rate * d_weights
        # cannot be: `biases[i] +=` due to numpy
        biases[i] = biases[i] + (learning_rate * d_biases)

        # Propagate the error backward
        if i <= 0:
            return d_error
        # if i > 0:
        d_error = np.dot(d_error, weights[i].T) * sigmoid_derivative(activations[i])
        return d_error

    def train_once_with_loss(self, input_val, expected_output_info, learning_rate, weights, biases):
        # Adjust input to handle the new expected_output format
        expected_output, is_negative_example = expected_output_info

        # Forward pass: Store activations for each layer
        activations = self.forward_pass(input_val, weights, biases)

        # Backward pass
        # Calculate output error using custom loss function
        output_error = np.array([
            self.custom_loss(activation, expected_output[i], is_negative_example)
            for i, activation in enumerate(activations[-1])
        ])
        total_error = np.sum(output_error ** 2) # maybe no **2 here because thats done in the custom_loss

        # output_error = expected_output - activations[-1]
        # total_error = np.sum(output_error ** 2)

        # Calculate gradient for output layer
        d_error = output_error * sigmoid_derivative(activations[-1])
        for i in reversed(range(len(weights))):
            d_error = self.calculate_current_layer(activations, i, d_error, learning_rate, weights, biases)

        # for i in reversed(range(len(weights))):
            # # Calculate error for the current layer
            # d_weights = np.outer(activations[i], d_error)
            # d_biases = d_error

            # # Update weights and biases
            # weights[i] += learning_rate * d_weights
            # # cannot be: `biases[i] +=` due to numpy
            # biases[i] = biases[i] + (learning_rate * d_biases)

            # # Propagate the error backward if not in the first layer
            # if i <= 0:
            #     continue
            # # if i > 0:
            # d_error = np.dot(d_error, weights[i].T) * sigmoid_derivative(activations[i])

        return total_error

    def custom_loss(self, predicted_output, true_output, is_negative_example):
        if is_negative_example:
            # Assuming the true_output for negative examples is what we want to avoid
            # This penalizes the model for predicting close to the true_output
            return 1 - abs(true_output - predicted_output)

        # Standard MSE loss for positive examples
        return true_output - predicted_output

    def forward_pass_one(self, input_data, weights, biases):
        # # Generalized forward pass through all layers
        # for i in range(len(weights)):
        #     input_data = np.dot(input_data, weights[i]) + biases[i]
        #     input_data = sigmoid(input_data)
        # return input_data
        return self.forward_pass(input_data, weights, biases)[-1]

    def forward_pass(self, input_data, weights, biases):
        activations = [input_data]
        for i in range(len(weights)):
            # on error: likely due to the input shape not matching the input nodes.
            input_data = np.dot(input_data, weights[i]) + biases[i]
            input_data = sigmoid(input_data)
            activations.append(input_data)
        return activations

    def predict_next_letter(self, input_val):
        return self.predict(letter_to_num[input_val],
                self.wb.weights,
                self.wb.biases,
                letter_to_num,
                num_to_letter)  # Predict next letter from 'a'

    def _expanded_predict(self, input_val, weights, biases, letter_to_num, num_to_letter):
        v = self.query(input_val, weights, biases)
        # Convert the final output to the closest letter
        # Here, we're assuming the output is a single number.
        # Adjust as needed for multiple outputs.
        final_output = v[0]  # Assuming single output for simplicity
        return self.closest_match(final_output, letter_to_num, num_to_letter)

    def query(self, values, weights=None, biases=None):
        # Ensure input is in the correct format (numpy array)
        input_val = convert_to_numpy(values)
        weights = weights or self.wb.weights
        biases = biases or self.wb.biases
        # Generalized forward pass through all layers
        res = self.forward_pass_one(input_val, weights, biases)
        return res

    def predict(self, value, weights=None, biases=None, translate_in=None, translate_out=None):
        val = self.query(value, weights, biases)
        translate_in = translate_in or letter_to_num
        translate_out = translate_out or num_to_letter
        return self.closest_match(val[0], translate_in, translate_out)

    def closest_match(self, value, value_in, value_out):
        closest_num = min(value_in.values(), key=lambda x: abs(x - value))
        predicted_letter = value_out[closest_num]
        return predicted_letter


def do_one(char, l, unit=None):
    v = (unit or nn).predict_next_letter(char)
    print(f'Predicted letter: {char} -> {v}, "{v == l}"')
    return v


def do_all(unit=None):
    unit = unit
    v = do_one('w', 'h', unit=unit)
    v = do_one('h', 'e', unit=unit)
    v = do_one('e', 'n', unit=unit)
    v = do_one('a', 't', unit=unit)


if __name__ == '__main__':
    main()