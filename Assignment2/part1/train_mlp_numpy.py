import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy, Linear

from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10
MODE_DEFAULT = 'batch'
BATCH_SIZE_DEFAULT = 32

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    predicted = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    correct_preds = np.sum(predicted == true_classes)
    accuracy = correct_preds / predictions.shape[0]
    return accuracy * 100


#def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, mode, batch_size):
def train(X_train, y_train, X_test, y_test, dnn_hidden_units, learning_rate, max_steps, eval_freq, mode, batch_size):
    print('using ' + mode + ' gradient descent')

    n_input = X_train.shape[1]
    n_hidden = list(map(int, dnn_hidden_units.split(',')))
    n_output = y_train.shape[1]

    model = MLP(n_input, n_hidden, n_output)
    crossEntropy = CrossEntropy()

    # for plotting performance purpose
    losses = []
    accuracies = []

    for step in range(max_steps):
        if mode == 'batch':
            predictions = model.forward(X_train)
            loss = crossEntropy.forward(predictions, y_train)
            grads = crossEntropy.backward(predictions, y_train)
            model.backward(grads)
            for layer in model.layers:
                if isinstance(layer, Linear):

                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias'] -= learning_rate * layer.grads['bias']

        elif mode == 'stochastic':
            # np.random.shuffle(training_indices)
            idx = np.random.randint(0, X_train.shape[0])
            X_train_stoch = X_train[idx, :].reshape(1, -1)
            y_train_stoch = y_train[idx, :].reshape(1, -1)
            predictions = model.forward(X_train_stoch)
            loss = crossEntropy.forward(predictions, y_train_stoch)
            grads = crossEntropy.backward(predictions, y_train_stoch)
            model.backward(grads)
            for layer in model.layers:
                if isinstance(layer, Linear):

                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias'] -= learning_rate * layer.grads['bias']

        elif mode == 'mini-batch':

            for i in range(0, X_train.shape[0], batch_size):
                X_train_batch = X_train[i:i + batch_size]
                y_train_batch = y_train[i:i + batch_size]
                predictions = model.forward(X_train_batch)
                loss = crossEntropy.forward(predictions, y_train_batch)
                grads = crossEntropy.backward(predictions, y_train_batch)
                model.backward(grads)
                for layer in model.layers:
                    if isinstance(layer, Linear):

                        layer.params['weight'] -= learning_rate * layer.grads['weight']
                        layer.params['bias'] -= learning_rate * layer.grads['bias']


        if step % eval_freq == 0 or step == max_steps - 1:
            test_predictions = model.forward(X_test)
            test_loss = crossEntropy.forward(test_predictions, y_test)
            test_accuracy = accuracy(test_predictions, y_test)  

            losses.append(test_loss)
            accuracies.append(test_accuracy)
            print(f"Step: {step}, Test Loss: {test_loss}, Accuracy: {test_accuracy}")

    print("Training complete!")
    return losses, accuracies


def load_data():
    X, y = make_moons(n_samples=1000, shuffle=True, noise=0.2, random_state=42)

    one_hot_encode = OneHotEncoder(sparse=False)
    y_onehot = one_hot_encode.fit_transform(y.reshape(-1,1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT, help='Batch size for mini-batch training')
    parser.add_argument('--mode', type=str, default=MODE_DEFAULT, choices=['batch', 'mini-batch', 'stochastic'],
                        help='Mode of training: batch, mini-batch, or stochastic')

    FLAGS = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data()

    losses, accuracies = train(X_train, y_train, X_test, y_test, FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.mode, FLAGS.batch_size)
    #train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.mode, FLAGS.batch_size)

if __name__ == '__main__':
    main()

