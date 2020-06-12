"""
Functions for training and evaluating the model
"""

import torch

import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score
import math
import time

from create_loader import *
from model_class import *
from plot import *

SEED = 412413

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CREATE = True
DATA_PATH = "../data/training_data/final_train_data.csv"

args = {
    'batch_size': 256,
    'lr': 3e-4,
    'hidden_dim': 128,
    'n_layers': 1,
    'bidirectional': True,
    'dropout': 0.2,
    'n_epochs': 20,
    'b1': 0.9,
    'b2': 0.999,
    'weight_decay': 0.01,
    'lr_decay': 0.7,
    'momentum': 0.9
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def multi_acc(y_pred, y_label):
    softmax = nn.Softmax(dim=1)
    y_pred_softmax = softmax(y_pred)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    # accu
    correct_pred = (y_pred_tags == y_label).float()
    acc = correct_pred.sum() / len(y_label)

    # f1
    f1 = f1_score(y_label.detach().cpu(), y_pred_tags.detach().cpu(), average='weighted')

    return acc, f1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, data_loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        predictions = model(data).squeeze(1)

        loss = criterion(predictions, target)

        acc, f1 = multi_acc(predictions, target)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1

        print("batch idx {}: | train loss: {} | train accu: {:.3f} | train f1: {}".format(
            batch_idx, loss.item(), acc.item(), f1))

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader), epoch_f1 / len(data_loader)


def evaluate(model, data_loader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            predictions = model(data).squeeze(1)
            loss = criterion(predictions, target)

            acc, f1 = multi_acc(predictions, target)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader), epoch_f1 / len(data_loader)


if __name__ == "__main__":
    if CREATE:
        train_loader, valid_loader, test_loader = create_loaders(args, DATA_PATH)
        torch.save(train_loader, "train_loader.pt")
        torch.save(valid_loader, "valid_loader.pt")
        torch.save(test_loader, "test_loader.pt")
    else:
        train_loader, valid_loader, test_loader = torch.load("train_loader.pt", "valid_loader.pt", "test_loader.pt")

    history = {
        "train_loss": [],
        "valid_loss": []
    }

    bert = RobertaModel.from_pretrained('roberta-base')
    model = BERTGRUSentiment(bert,
                             args['hidden_dim'],
                             3,
                             args['n_layers'],
                             args['bidirectional'],
                             args['dropout'])

    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False

    print(f'The model has {count_parameters(model):,} trainable parameters')

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=args["lr_decay"])
    criterion = nn.CrossEntropyLoss(weight=args['weight']).to(device)

    best_valid_loss = float('inf')

    # the main training loop
    for epoch in range(args['n_epochs']):

        start_time = time.time()

        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)
        history["train_loss"].append(train_loss)
        valid_loss, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion)
        history["valid_loss"].append(valid_loss)
        scheduler.step()

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f} | Train f1: {train_f1}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f} | Val. f1: {valid_f1}%')

    # evaluation    
    model.load_state_dict(torch.load('best_model.pt'))
    valid_loss, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion)
    print("Valid loss: {} | Valid Acc: {:.3f} |  Valid f1: {}".format(
        valid_loss, valid_acc, valid_f1))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    print("Test loss: {} | Test Acc: {:.3f} | Test f1: {}".format(
        test_loss, test_acc, test_f1))

    plot_history(history)
