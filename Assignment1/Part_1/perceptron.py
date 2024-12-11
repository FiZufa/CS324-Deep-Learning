import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.01):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(n_inputs + 1)  # Fill in: Initialize weights with zeros 
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returnsa:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        predict = np.dot(input_vec, self.weights[1:]) + self.weights[0] # y^ = wT.x + b
        # print(f"Input: {input_vec}, Activation: {activation}")
        return 1 if predict >= 0 else -1
        
    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        self.training_accuracy = []
        # we need max_epochs to train our model
        for _ in range(self.max_epochs): 
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            errors = 0
            weight_updates = np.zeros(self.n_inputs + 1)
            
            for inputs, label in zip(training_inputs, labels):
                predicted_y = self.forward(inputs)

                if label * predicted_y <= 0: # if prediction is incorrect
                    errors += 1
                    weight_updates[1:] +=  label * inputs
                    weight_updates[0] += label

            # update weights
            self.weights[1:] += self.learning_rate * weight_updates[1:]
            self.weights[0] += self.learning_rate * weight_updates[0]

            y_pred = np.array([self.forward(x) for x in training_inputs])
            accuracy = np.mean(y_pred == labels)
            print(f'Epoch {_}: Training Accuracy: {accuracy:.2f}, Loss: {errors}')
            self.training_accuracy.append(accuracy)
