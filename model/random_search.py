"""
Using random search to do hyperparameter tuning
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

from create_loader import *
from model_class import *
from plot import *
from train_model import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_EVALS = 1
output_filename = "./search_result/search.csv"


def run_model(space):
    args = {'batch_size': 128,
            'lr': space['lr'],
            'hidden_dim': 128,
            'n_layers': space['n_layers'],
            'bidirectional': True,
            'dropout': space['dropout'],
            'n_epochs': 20,
            'b1': space['b1'],
            'b2': space['b2'],
            'weight_decay': space['weight_decay'],
            'weight': torch.tensor([0.1568, 0.4639, 0.3793], dtype=torch.float32)
            }
    train_loader = torch.load("train_loader.pt")
    valid_loader = torch.load("valid_loader.pt")
    test_loader = torch.load("test_loader.pt")

    opt_name = '_'.join(['b1_' + str(args['b1']), 'b2_' + str(args['b2']), 'lr' + str(args['lr']),
                         'drop' + str(args['dropout']), 'l2_' + str(args['weight_decay'])])

    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = BERTGRUSentiment(bert,
                             args['hidden_dim'],
                             3,
                             args['n_layers'],
                             args['bidirectional'],
                             args['dropout']).to(device)
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False
    optimizer = optim.Adam(model.parameters(),
                           lr=args['lr'],
                           betas=(args["b1"], args["b2"]),
                           weight_decay=args["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=args['weight']).to(device)
    history = {
        "train_loss": [],
        "valid_loss": []
    }

    best_valid_loss = float('inf')
    best_valid_acc = 0
    best_valid_f1 = 0

    for epoch in range(args['n_epochs']):

        start_time = time.time()

        train_loss, train_acc, train_rocauc, train_f1 = train(model, train_loader, optimizer, criterion)
        history["train_loss"].append(train_loss)
        valid_loss, valid_acc, valid_rocauc, valid_f1 = evaluate(model, valid_loader, criterion)
        history["valid_loss"].append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_valid_f1 = valid_f1

    model_dict = {
        "v_loss": best_valid_loss,
        "v_acc": best_valid_acc,
        "v_f1": best_valid_f1,
        "name": opt_name
    }
    return model_dict


if __name__ == "__main__":
    # hyper parameter search space
    space = {
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(3e-2)),
        'n_layers': hp.choice("n_layers", range(2, 4, 1)),
        'dropout': hp.uniform("dropouut", 0.25, 0.5),
        'b1': hp.loguniform('b1', np.log(0.5), np.log(0.9)),
        'b2': hp.loguniform('b2', np.log(0.5), np.log(0.999)),
        'weight_decay': hp.loguniform('weight_decay', np.log(0.01), np.log(1))
    }

    trials = Trials()
    evals_inc = min(NUM_EVALS, 1)
    while evals_inc <= NUM_EVALS:
        best = fmin(fn=run_model, space=space, algo=tpe.suggest, max_evals=evals_inc,
                    trials=trials)
        results = []
        for trial in trails.trails:
            results.append(trail['result'])
        keys = results[0].keys()
        with(open(output_filename, "w")) as output_file:
            dict_writer = csv.DictWriter(output_filename, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        if evals_inc == NUM_EVALS:
            break
        evals_inc = min(NUM_EVALS, evals_inc + 5)
