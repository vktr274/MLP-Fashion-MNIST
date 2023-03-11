from torch.utils.data import DataLoader, random_split
from math import floor
from torch import argmax
import wandb
import torch
from torch.nn import Sequential, Linear, ReLU, LogSoftmax, Flatten


def split_data(data, test, validation_split=0.2, batch_size=32):
    data_length = len(data)

    validation_size = floor(validation_split * data_length)

    validation, train = random_split(
        data, [validation_size, data_length - validation_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def fit(model, train_loader, val_loader, optimizer, loss_fn, epochs, batch_size):
    history = {
        'val_loss': [],
        'val_accuracy': [],
        'loss': [],
        'accuracy': []
    }
    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0
        train_loss = 0
        val_loss = 0

        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            y_pred_dist = model(x_train)
            y_pred = argmax(y_pred_dist, dim=1)
            loss = loss_fn(y_pred_dist, y_train)

            train_loss += loss.item()
            train_total += len(y_train)
            train_correct += (y_pred == y_train).sum().item()
            loss.backward()
            optimizer.step()

        for x_val, y_val in val_loader:
            with torch.no_grad():
                y_pred_dist = model(x_val)
                y_pred = argmax(y_pred_dist, dim=1)
                loss = loss_fn(y_pred_dist, y_val)
                val_loss += loss.item()
                val_total += len(y_val)
                val_correct += (y_pred == y_val).sum().item()

        val_loss /= len(val_loader.dataset)/batch_size
        train_loss /= len(train_loader.dataset)/batch_size

        val_acc = val_correct/val_total
        train_acc = train_correct/train_total

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        wandb.log({
            'loss': train_loss,
            'val_loss': val_loss,
            'accuracy': train_acc,
            'val_accuracy': val_acc,
        })

        print(f'Epoch {epoch + 1}/{epochs}: loss {train_loss}, val_loss {val_loss}, ' +
              f'accuracy {train_acc}, val_accuracy {val_acc}')

    return history


def create_model(config):
    mlp = Sequential(
        Flatten(),
        Linear(28 * 28, config.dense_1),
        ReLU(),
        Linear(config.dense_1, config.dense_2),
        ReLU(),
        Linear(config.dense_2, config.dense_3),
        ReLU(),
        Linear(config.dense_3, config.dense_4),
        ReLU(),
        Linear(config.dense_4, 10),
        LogSoftmax(dim=1)
    )
    return mlp


def evaluate(model, test_loader):
    y_test_all = []
    y_pred_all = []
    total = 0
    correct = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            outputs = model(x_test)
            predicted = argmax(outputs, dim=1)
            y_pred_all.extend(predicted.numpy())
            y_test_all.extend(y_test.numpy())
            total += len(y_test)
            correct += (predicted == y_test).sum().item()

    return y_test_all, y_pred_all
