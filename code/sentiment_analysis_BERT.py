import numpy as np
import pandas as pd
import torch
import unidecode
import transformers as ppb  # pytorch transformers
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# text = "I'd like to have three cups   of coffee<br /><br />from your Café. #delicious"
# result = text_preprocessing(text)
# print(result)
# # result: ['like', 'cup', 'coffee', 'cafe', 'delicious']

# from text_preprocess import text_preprocessing
# contents = df.values[:, 2]
# for i in range(len(contents)):
#     contents[i] = text_preprocessing(contents[i])
#     print(contents[i])

if __name__ == "__main__":
    # df[0]: id, df[1]: target, df[2]: content
    df = pd.read_csv(
        'https://raw.githubusercontent.com/cbaziotis/datastories-semeval2017-task4/master/dataset/Subtask_A'
        '/downloaded/twitter-2013test-A.tsv',
        delimiter='\t',
        header=None)

    # import model
    model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # load pretrained model and weights
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # list of sentences
    tokenized = df[2].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # pad all lists to the same size as a 2d-array: so BERT can process once not many times
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

    # set attentions at padded positions to 0
    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids.type(torch.LongTensor), attention_mask=attention_mask)

    # logistics regression model
    features = last_hidden_states[0][:, 0, :].numpy()
    labels = df[1]
    labels = labels.replace(to_replace='positive', value=1)
    labels = labels.replace(to_replace='negative', value=-1)
    labels = labels.replace(to_replace='neutral', value=0)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    # grid search
    parameters = {'max_iter': [1000], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15),
                  'random_state':[0,1,2,3,4,5,6,7,8,9]}
    grid_search = GridSearchCV(MLPClassifier(), parameters)
    grid_search.fit(train_features, train_labels)

    print('best parameters: ', grid_search.best_params_)
    print('best scores: ', grid_search.best_score_)