from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import RMSprop
import torch.nn.functional as F

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy


def train(model, data_loader, optimizer, criterion, max_norm, device):
    # TODO set model to train mode
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_norm)
        
        acc = accuracy(F.softmax(outputs, dim=1), batch_targets)
        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))
        # Add more code here ...
        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)

    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    # TODO set model to evaluation mode
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        acc = accuracy(F.softmax(outputs, dim=1), batch_targets)
        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))

        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def main(input_length, input_dim, num_classes, num_hidden, batch_size, learning_rate, max_epoch, max_norm, data_size, portion_train):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the model that we are going to use
    model = VanillaRNN(input_dim=input_dim, hidden_dim=num_hidden, output_dim=num_classes, input_length=input_length)
    model.to(device)

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(input_length, data_size)

    train_size = int(len(dataset) * portion_train)
    val_size = len(dataset) - train_size

    # Split dataset into train and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in range(max_epoch):
        # Train the model for one epoch
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, max_norm, device)

        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device)
        
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Accuracy {train_acc:.2f}, "
              f"Val Loss {val_loss:.4f}, Val Accuracy {val_acc:.2f}")
        
        scheduler.step()

    print('Done training.')
    return train_accuracies, train_losses, val_accuracies, val_losses

    


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=19,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=1000, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=1000000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(**vars(config))
