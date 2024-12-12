from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 50  
EVAL_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './data'

FLAGS = None

def evaluate(net, test_loader, epoch, loss_function, device):
    net.eval()
    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    avg_loss = total_loss / total
    print(f'Epoch: {epoch+1}, Test Accuracy: {acc:.2f}, Test Loss: {avg_loss:.2f}')
    return acc, avg_loss


def train(net, train_loader, optimizer, loss_function, epoch, batch, device):
    net.train()
    total_loss = 0.0
    correct = 0
    total = 0
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % batch == 0:
            # print(f'Batch {i + 1}, Loss: {running_loss / batch:.2f}')
            running_loss = 0.0

    acc = correct / total
    avg_loss = total_loss / total
    print(f'Epoch: {epoch+1}, Train Accuracy: {acc:.2f}, Test Loss: {avg_loss:.2f}')
    return acc, avg_loss

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return 100 * correct / labels.size(0)

def train_and_evaluate(trainloader, testloader, learning_rate, max_epochs, eval_freq, batch_size, selected_optim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(n_channels=3, n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if selected_optim == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(f'Optimizer: {selected_optim}')
    elif selected_optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        print(f'Optimizer: {selected_optim}')

    train_accuracy = []
    test_accuracy = []
    train_losses = []
    test_losses = []

    for epoch in range(max_epochs):
        train_acc, train_loss = train(net=model, train_loader=trainloader,
                                      optimizer=optimizer, loss_function=criterion,
                                      epoch=epoch, batch=batch_size, device=device)
        train_accuracy.append(train_acc)
        train_losses.append(train_loss)

        test_acc, test_loss = evaluate(net=model, test_loader=testloader,
                                       epoch=epoch, loss_function=criterion, device=device)
        test_accuracy.append(test_acc)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch+1}, Train Accuracy: {train_acc:.2f}, Train Loss: {train_loss:.2f}')
        print(f'Epoch: {epoch+1}, Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.2f}')
    
    return train_accuracy, train_losses, test_accuracy, test_losses

def main():
    """
    Main function
    """
    train_and_evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    main()
