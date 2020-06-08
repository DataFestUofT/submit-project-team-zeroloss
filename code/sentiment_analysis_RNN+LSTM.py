import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch transformers
from collections import Counter
from string import punctuation
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from torchtext.vocab import GloVe
from torchtext.data import Field
EMBEDDING_DIM = 100
embedding_dict = GloVe(name='twitter.27B', dim=EMBEDDING_DIM)

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
# from keras.utils.np_utils import to_categorical
# from keras.callbacks import EarlyStopping
# from keras.layers import Dropout

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

def pad_features(c_int, seq_length):
    ''' Return features of intergerized contents, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(c_int), seq_length), dtype=int)

    for i, review in enumerate(c_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            new = zeroes + review
        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i, :] = np.array(new)
    return features


# LSTM Class
class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        # self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        # embeds = self.embedding(x)
        # lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out, hidden = self.lstm(x.float(), hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # reshape to be batch_size first
        out = out.view(output_size*batch_size, -1)
        out = out[:, -1]  # get last batch of labels

        out = out.view(batch_size, output_size)

        # return last sigmoid output and hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden


if __name__ == "__main__":
    # df[0]: id, df[1]:tweet_id, df[2]: sentiment, df[3]: content
    df1 = pd.read_csv(
        '../data/training_data/data1.csv',
        sep=',')
    df2 = pd.read_csv(
        '../data/training_data/data2.csv',
        sep=',')
    # df3 = pd.read_csv(
    #     '../data/training_data/data3.csv',
    #     sep=',')
    df = df1.append(df2, ignore_index=True)
    # df = df.append(df3, ignore_index=True)

    # text preprocess
    # uncased
    df["text"] = df["text"].str.lower()
    # remove punctuation
    escaped_punct = punctuation.replace('$', '')
    escaped_punct = escaped_punct.replace('\'', '')
    escaped_punct = escaped_punct.replace('[', '')
    escaped_punct = escaped_punct.replace(']', '')
    contents = df['text'].str.replace('[{}]'.format(escaped_punct), '')
    contents = contents.str.replace("'s", " 's")

    # padding
    # find max length
    max_len = 0
    for i in contents:
        if len(i.split()) > max_len:
            max_len = len(i)

    # pretrained GloVe:
    features = torch.zeros([contents.size, max_len, EMBEDDING_DIM], dtype=torch.float32)
    for i in range(contents.size):
        contents[i] = contents[i].split()
    for tweet_idx in range(contents.size):
        tweet = contents[tweet_idx]
        for word_idx in range(len(tweet)):
            word_vec = embedding_dict[tweet[word_idx]].float()
            features[tweet_idx][word_idx] = word_vec

    # labels
    labels = df['sentiment']
    labels = labels.replace(to_replace='positive', value=2)
    labels = labels.replace(to_replace='neutral', value=1)
    labels = labels.replace(to_replace='negative', value=0)

    # prepare datasets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.1)
    train_features, valid_features, train_labels, valid_labels = train_test_split(train_features, train_labels, test_size=0.11)
    # create Tensor datasets
    train_data = TensorDataset(train_features, torch.from_numpy(train_labels.values))
    test_data = TensorDataset(test_features, torch.from_numpy(test_labels.values))
    valid_data = TensorDataset(valid_features, torch.from_numpy(valid_labels.values))
    # dataloaders
    batch_size = 100
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)

    vocab_size = 25000  # +1 for the 0 padding
    output_size = 3
    hidden_dim = 256
    n_layers = 3
    net = SentimentLSTM(vocab_size, output_size, EMBEDDING_DIM, hidden_dim, n_layers).to(device)
    print(net)

    # loss and optimization functions
    lr = 1e-4
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.15, 0.48, 0.37])).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params
    epochs = 50
    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    best_valid_loss = float('inf')
    los = []
    val_los = []

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            # inputs = inputs.long()
            inputs, labels = inputs.to(device), labels.to(device)
            output, h = net(inputs, h)
            # calculate the loss and perform backprop
            loss = criterion(output, labels)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    inputs = inputs.long()
                    inputs, labels = inputs.to(device), labels.to(device)
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, labels)
                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

                los.append(loss.item())
                val_los.append(np.mean(val_losses))

                if best_valid_loss > np.mean(val_losses):
                    best_valid_loss = np.mean(val_losses)
                    torch.save(net.state_dict(), 'best-model.pt')

    plt.plot(range(len(los)), los)
    plt.plot(range(len(los)), val_los)

    net.load_state_dict(torch.load('best-model.pt'))
    # Get test data loss and accuracy
    test_losses = []  # track loss
    num_correct = 0
    # init hidden state
    h = net.init_hidden(batch_size)
    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        # get predicted outputs
        inputs = inputs.long()
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = net(inputs, h)
        # calculate loss
        test_loss = criterion(output, labels)
        test_losses.append(test_loss.item())
        # convert output probabilities to predicted class (0 or 1)
        temp = torch.round(output)  # rounds to the nearest integer
        with torch.no_grad():
            temp = temp.cpu().numpy()
        pred = []
        for out in temp:
            idx = out.argmax()
            pred.append(idx)
        pred = torch.FloatTensor(pred).to(device)

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))