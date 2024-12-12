from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pytorch_mlp import MLP

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
WEIGHT_DECAY_DEFAULT = 1e-5  
FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions of the network.
    Args:
        predictions: 2D float tensor of size [batch_size, num_classes]
        targets: 1D long tensor of size [batch_size] with class indices
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    predicted_classes = predictions.argmax(dim=1)  # Get the index of the max logit which is the predicted class
    correct_preds = (predicted_classes == targets).sum()  # Compare with targets which should already be indices
    accuracy = 100.0 * correct_preds.float() / targets.size(0)
    return accuracy


def train(X_train, y_train, X_test, y_test, dnn_hidden_units, learning_rate, max_steps, eval_freq, weight_decay):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should evaluate the model on the whole test set each eval_freq iterations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert numpy arrays to tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)
    
    n_input = X_train.shape[1]
    n_hidden = list(map(int, dnn_hidden_units.split(',')))
    n_output = len(torch.unique(y_train))

    model = MLP(n_input, n_hidden, n_output).to(device)

    if y_train.dtype != torch.float32:
        y_train = y_train.long()
    if y_test.dtype != torch.float32:
        y_test = y_test.long()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    losses = []
    accuracies = []

    for step in range(max_steps):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.long())  
        loss.backward()
        optimizer.step()

        if step % eval_freq == 0 or step == max_steps - 1:
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                test_loss = criterion(outputs, y_test.long()) 
                acc = accuracy(torch.softmax(outputs, dim=1), y_test)  
                losses.append(test_loss.item())
                accuracies.append(acc.item())
                print(f"Step: {step}, Test Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%")

    print("Training complete!")
    return losses, accuracies

        

def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()