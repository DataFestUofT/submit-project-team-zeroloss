"""
Tokenize the texts and create train, valid, test dataloaders 
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

# Define the dataset
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, x, labels):
        'Initialization'
        self.x = x
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return self.x.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        x = self.x[index]
        y = self.labels[index]

        return x, y

def get_weight(df, args):
    frac_positive = (df["sentiment"] == "positive").sum() / len(df)
    frac_negative = (df["sentiment"] == "negative").sum() / len(df)
    frac_neutral = (df["sentiment"] == "neutral").sum() / len(df)

    args["weight"] = torch.tensor([frac_positive, frac_negative, frac_neutral], dtype=torch.float32)

def get_max_len(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    return max_len
    
def create_loaders(args, data_path):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1)
    get_weight(df, args)
    train_num = math.ceil(0.7 * len(df))
    valid_num = math.ceil(0.9 * len(df))
    train_data = df.iloc[:train_num, :].reset_index()
    valid_data = df.iloc[train_num:valid_num, :].reset_index()
    test_data = df.iloc[valid_num:, :].reset_index()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    tokenized_train = train_data['text'].apply((
        lambda x: tokenizer.encode(x, add_special_tokens=True)))
    tokenized_valid = valid_data['text'].apply((
        lambda x: tokenizer.encode(x, add_special_tokens=True)))
    tokenized_test = test_data['text'].apply((
        lambda x: tokenizer.encode(x, add_special_tokens=True)))
    
    max_len_train = get_max_len(tokenized_train)
    max_len_valid = get_max_len(tokenized_valid)
    max_len_test = get_max_len(tokenized_test)
    max_len = max([max_len_train, max_len_valid, max_len_test])

    padded_train = torch.tensor([i + [0] * (max_len - len(i)) 
                                for i in tokenized_train.values])
    padded_valid = torch.tensor([i + [0] * (max_len - len(i)) 
                                for i in tokenized_valid.values])
    padded_test = torch.tensor([i + [0] * (max_len - len(i)) 
                                for i in tokenized_test.values])

    train_label = torch.tensor(train_data['sentiment'].replace(
        to_replace='positive', value=2).replace(
        to_replace='negative', value=0).replace(
        to_replace='neutral', value=1))
    valid_label = torch.tensor(valid_data['sentiment'].replace(
        to_replace='positive', value=2).replace(
        to_replace='negative', value=0).replace(
        to_replace='neutral', value=1))
    test_label = torch.tensor(test_data['sentiment'].replace(
        to_replace='positive', value=2).replace(
        to_replace='negative', value=0).replace(
        to_replace='neutral', value=1))

    trainset = Dataset(padded_train, train_label)
    validset = Dataset(padded_valid, valid_label)
    testset = Dataset(padded_test, test_label)

    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args['batch_size'],
                                            shuffle=True,
                                            drop_last=True)
    valid_loader = torch.utils.data.DataLoader(validset,
                                            batch_size=args['batch_size'],
                                            shuffle=True,
                                            drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                            batch_size=args['batch_size'],
                                            shuffle=True,
                                            drop_last=True)

    return train_loader, valid_loader, test_loader