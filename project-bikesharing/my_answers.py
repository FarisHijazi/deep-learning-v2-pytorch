import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.0)


debug = True


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes + 1, self.hidden_nodes + 1))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes + 1, self.output_nodes))
        self.lr = learning_rate

        #### : Set self.activation_function to your implemented sigmoid function ####

        # this contains both the activation function and its derivative
        self.activation_function = sigmoid  # Replace 0 with your sigmoid calculation.

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''

        if debug:
            print("features.shape=", features.shape)
            print("weights_input_to_hidden.shape=", self.weights_input_to_hidden.shape)
            print("weights_hidden_to_output.shape=", self.weights_hidden_to_output.shape)

        features = np.hstack((np.ones((features.shape[0], 1)), features))
        n_records = features.shape[0]

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch
        '''
        ### Forward pass ###

        # : Hidden layer
        # signals into hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # : Output layer -
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        '''
        ## # this code is from the excercises
        ## output_error_term = error * sigmoid_prime(h)
        ## hidden_error = np.dot(output_error_term, weights_hidden_output)
        ## hidden_error_term = hidden_error * sigmoid_prime(hidden_input)
        #
        # >> output_error_term ()
        # >> hidden_error_term (2,)
        # >> correction_i_h: (6, 2)
        # >> correction_h_o: (2,)

        ### Backward pass ###

        # : Output error
        error = y - final_outputs  # Output layer error is the difference between desired target and actual output.

        # in this line only, I had a look at this solution: https://github.com/absalomhr/Predicting-Bike-Sharing-Patterns/blob/master/my_answers.py
        hidden_error = np.dot(self.weights_hidden_to_output, error)

        # : Backpropagated error terms -
        # f_prime_final = (final_outputs * (1 - final_outputs))
        output_error_term = error

        # : Calculate the hidden layer's contribution to the error
        f_prime_hidden = (hidden_outputs * (1 - hidden_outputs))
        hidden_error_term = hidden_error * f_prime_hidden

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        # #### Implement the forward pass here ####
        # exactly the same as the forward_pass_train()
        final_outputs, hidden_inputs = self.forward_pass_train(features)
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1000
learning_rate = 0.5
hidden_nodes = 25
output_nodes = 1
