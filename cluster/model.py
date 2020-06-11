# coding: utf-8

# In[1]:


import torch

import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score, roc_auc_score

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-bs', '--batchsize', action='store', default=128,
                    dest="bs", type=int, help="The batch size to use")

parser.add_argument('-lr', '--learningrate', action="store", default=7e-3,
                    dest="lr", type=float, help="The learning rate")

parser.add_argument('-hd', '--hiddendim', action="store", default=128,
                    dest="hd", type=int, help="The hidden layer dimensionality")

parser.add_argument('-nl', '--nlayers', action="store", default=3,
                    dest="nl", type=int, help="The number of layers to use for NN")

parser.add_argument('-do', '--dropout', action="store", default=0.25,
                    dest="do", type=float, help="The probability of dropout")

parser.add_argument('-ne', '--nepochs', action="store", default=70,
                    dest="ne", type=int, help="The number of training epochs")

parser.add_argument('-b1', '--beta1', action="store", default=0.9,
                    dest="b1", type=float, help="beta 1 to use for adam and adamw")

parser.add_argument('-b2', '--beta2', action="store", default=0.999,
                    dest="b2", type=float, help="beta 2 to use for adam and adamw")

parser.add_argument('-wd', '--weightdecay', action="store", default=0.01,
                    dest="wd", type=float, help="weight decay factor for adam")

parser.add_argument('-sn', '--save-name', action="store", dest="save_name", 
                    help="name for saving model and graph")

cmd_args = parser.parse_args()


args = {
    'batch_size': cmd_args.bs,
    'lr': cmd_args.lr,
    'hidden_dim': cmd_args.hd,
    'n_layers': cmd_args.nl,
    'bidirectional': True,
    'dropout': cmd_args.do,
    'n_epochs': cmd_args.ne,
    'b1': cmd_args.b1,
    'b2': cmd_args.b2,
    'weight_decay': cmd_args.wd,
    'lr_decay': 0.7
}


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[2]:


df = pd.read_csv("./data/training_data/final_train_data.csv", sep=",")

import math

df = df.sample(frac=1)

train_num = math.ceil(0.7 * len(df))
valid_num = math.ceil(0.9 * len(df))
train_data = df.iloc[:train_num, :].reset_index()
valid_data = df.iloc[train_num:valid_num, :].reset_index()
test_data = df.iloc[valid_num:, :].reset_index()


# In[3]:


df.head()


# In[4]:

num_positive = (df["sentiment"] == "positive").sum()
num_negative = (df["sentiment"] == "negative").sum()
num_neutral = (df["sentiment"] == "neutral").sum()

args["weight"] = torch.tensor([num_negative / len(df), num_neutral / len(df), num_positive / len(df)], dtype=torch.float32)

print(args["weight"])


# In[5]:


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# In[6]:


tokenized_train = train_data['text'].apply((
    lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized_valid = valid_data['text'].apply((
    lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized_test = test_data['text'].apply((
    lambda x: tokenizer.encode(x, add_special_tokens=True)))


# In[7]:


def get_max_len(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    return max_len


# In[8]:


max_len_train = get_max_len(tokenized_train)
print(max_len_train)
max_len_valid = get_max_len(tokenized_valid)
print(max_len_valid)
max_len_test = get_max_len(tokenized_test)
print(max_len_test)
max_len = max([max_len_train, max_len_valid, max_len_test])


# In[9]:


padded_train = torch.tensor([i + [0] * (max_len - len(i)) 
                             for i in tokenized_train.values])
padded_valid = torch.tensor([i + [0] * (max_len - len(i)) 
                             for i in tokenized_valid.values])
padded_test = torch.tensor([i + [0] * (max_len - len(i)) 
                            for i in tokenized_test.values])


# In[10]:


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


# In[11]:


# Define the dataset and data iterators
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


# In[12]:


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


# In[13]:


# torch.save(trainset, "trainset.pt")
# torch.save(validset, "validset.pt")
# torch.save(testset, "testset.pt")


# In[14]:


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
#         embedding_dim = bert.config.to_dict()['dim']
        embedding_dim = 768
    
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
        attention_mask = text.masked_fill(text != 0, 1)
                
        with torch.no_grad():
            embedded = self.bert(text, attention_mask=attention_mask)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output



class BERTLSTMSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
#         embedding_dim = bert.config.to_dict()['dim']
        embedding_dim = 768
    
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers = n_layers,
                           bidirectional = bidirectional,
                           batch_first = True,
                           dropout = 0 if n_layers < 2 else dropout)

        self.linear1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 
                                 hidden_dim * 2 if bidirectional else hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
        attention_mask = text.masked_fill(text != 0, 1)
                
        with torch.no_grad():
            embedded = self.bert(text, attention_mask=attention_mask)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, (hidden, _) = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.tanh(self.linear2(self.tanh(self.linear1(hidden))))
        
        #output = [batch size, out dim]
        
        return output

# In[15]:


bert = RobertaModel.from_pretrained('roberta-base')


# In[16]:

# model = BERTGRUSentiment(bert,
#                          args['hidden_dim'],
#                          3,
#                          args['n_layers'],
#                          args['bidirectional'],
#                          args['dropout'])

model = BERTLSTMSentiment(bert,
                          args['hidden_dim'],
                          3,
                          args['n_layers'],
                          args['bidirectional'],
                          args['dropout'])

model = model.to(device)


# In[17]:


for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False


# In[18]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[19]:


for name, param in model.named_parameters():                
    if param.requires_grad:
        print(name)


# In[31]:


# optimizer = optim.Adam(model.parameters(), 
#                        lr=args['lr'], 
#                        betas=(args["b1"], args["b2"]),
#                        weight_decay=args["weight_decay"])

optimizer = optim.AdamW(model.parameters(), 
                       lr=args['lr'], 
                       betas=(args["b1"], args["b2"]),
                       weight_decay=args["weight_decay"])

# optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args["lr"])

scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=args["lr_decay"])

criterion = nn.CrossEntropyLoss(weight=args['weight']).to(device)


# In[21]:


def multi_acc(y_pred, y_label):
    softmax = nn.Softmax(dim=1)
    y_pred_softmax = softmax(y_pred)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
#     print(y_pred_tags)

    # accu
    correct_pred = (y_pred_tags == y_label).float()
    acc = correct_pred.sum() / len(y_label)

    # roc-auc
    # one_hot_label = nn.functional.one_hot(y_label)
    # roc_auc = roc_auc_score(one_hot_label.detach().cpu(), y_pred_softmax.detach().cpu(), average="macro")
    roc_auc = 1.

    # f1
    f1 = f1_score(y_label.detach().cpu(), y_pred_tags.detach().cpu(), average='weighted')
    
    return acc, roc_auc, f1


# In[22]:


def train(model, data_loader, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_rocauc = 0
    epoch_f1 = 0
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(data).squeeze(1)
        
        loss = criterion(predictions, target)
        
        acc, roc_auc, f1 = multi_acc(predictions, target)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_rocauc += roc_auc
        epoch_f1 += f1

        # print("batch idx {}: | train loss: {} | train accu: {:.3f} | train roc: {:.3f} | train f1: {}".format(
        #     batch_idx, loss.item(), acc.item(), roc_auc, f1))
        
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader), epoch_rocauc / len(data_loader), epoch_f1 / len(data_loader)


# In[23]:


def evaluate(model, data_loader, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_rocauc = 0
    epoch_f1 = 0
    model.eval()
    
    with torch.no_grad():
    
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            predictions = model(data).squeeze(1)
            
            loss = criterion(predictions, target)
            
            acc, roc_auc, f1 = multi_acc(predictions, target)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_rocauc += roc_auc
            epoch_f1 += f1
        
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader), epoch_rocauc / len(data_loader), epoch_f1 / len(data_loader)


# In[24]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:


history = {
    "train_loss": [],
    "valid_loss": []
}

import time

best_valid_loss = float('inf')

for epoch in range(args['n_epochs']):
    
    start_time = time.time()
    
    train_loss, train_acc, train_rocauc, train_f1 = train(model, train_loader, optimizer, criterion)
    history["train_loss"].append(train_loss)
    valid_loss, valid_acc, valid_rocauc, valid_f1 = evaluate(model, valid_loader, criterion)
    history["valid_loss"].append(valid_loss)
    scheduler.step()
        
    end_time = time.time()
        
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f"{cmd_args.save_name}.pt")
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Train rocauc: {train_rocauc} | Train f1: {train_f1}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f} | Val. rocauc: {valid_rocauc} | Val. f1: {valid_f1}%')


# In[26]:


model.load_state_dict(torch.load(f"{cmd_args.save_name}.pt"))


# In[27]:


valid_loss, valid_acc, valid_rocauc, valid_f1 = evaluate(model, valid_loader, criterion)
print("Valid loss: {} | Valid Acc: {:.3f} | Valid ROC-AUC: {} | Valid f1: {} | model: {}".format(
    valid_loss, valid_acc, valid_rocauc, valid_f1, cmd_args.save_name))
test_loss, test_acc, test_rocauc, test_f1 = evaluate(model, test_loader, criterion)
print("Test loss: {} | Test Acc: {:.3f} | Test ROC-AUC: {} | Test f1: {} | model: {}".format(
    test_loss, test_acc, test_rocauc, test_f1, cmd_args.save_name))


# In[28]:

import matplotlib.pyplot as plt

def plot_history(hist):
    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(1, len(history["train_loss"]) + 1), history["train_loss"], label="training loss")
    plt.plot(np.arange(1, len(history["train_loss"]) + 1), history["valid_loss"], label="validation loss")
    plt.legend(loc="best")
    plt.title("Training and Validation Losses")
    plt.savefig(f"{cmd_args.save_name}.png")
    # plt.show()


# In[30]:
plot_history(history)