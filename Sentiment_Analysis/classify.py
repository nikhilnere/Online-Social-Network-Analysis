"""
classify.py
"""


import json
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import configparser

def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


pos_words = []
neg_words = []

def read_afinnwords ():

    file = open('afinnwords.txt', 'r')
    afinn_dict = json.loads(file.read())
    for k,v in afinn_dict.items():
        if v < 0:
            neg_words.append(k)
        else:
            pos_words.append(k)

def tokenize(doc, keep_internal_punct=False):
    if (keep_internal_punct):
        tokens = []
        for term in doc.lower().split(' '):
            token = re.sub('^\W+', '', re.sub('\W+$', '', term)).strip()
            #tokens.append(re.sub('^\W+', '', re.sub('\W+$', '', term)))
            if (token):
                tokens.append(token)
        #tokens = [re.sub('^\W+', '', re.sub('\W+$', '', term)) for term in doc.lower().split(' ')]
        return np.array(tokens)
    else:
        tokens = re.sub('\W+', ' ', doc.lower()).split()
        return np.array(tokens)
    pass


def token_features(tokens, feats):
    for token in tokens:
        key = 'token=' + token
        if (feats.__contains__(key)):
            feats[key] = feats[key] + 1
        else:
            feats[key] = 1
    pass


def token_pair_features(tokens, feats, k=3):
    i = 0
    for i in range(0, tokens.size - k + 1):
        for j in range(i, i+k):
            for l in range (j+1, i+k):
                #print (str(j) + ' ' + str(l))
                key = 'token_pair=' + tokens[j] + '__' + tokens[l]
                # (key)
                if (feats.__contains__(key)):
                    feats[key] = feats[key] + 1
                else:
                    feats[key] = 1
    pass

def lexicon_features(tokens, feats):
    neg = 0
    pos = 0
    for token in tokens:
        if neg_words.__contains__(token.lower()):
            neg = neg + 1
        if pos_words.__contains__(token.lower()):
            pos = pos + 1

    feats['neg_words'] = neg
    feats['pos_words'] = pos
    pass


def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for fun in feature_fns:
        fun(tokens, feats)
    return sorted(feats.items(), key=lambda x:x[0])

    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    #Creating vocab first
    featslist = []
    for doc in tokens_list:
        featslist.append(featurize(doc,feature_fns))

    if vocab == None:
        alltokens = []
        for feats in featslist:
            alltokens.extend([tup[0] for tup in feats if tup[1] > 0])

        vocaball = dict()
        for token in alltokens:
            if vocaball.__contains__(token):
                vocaball[token] = vocaball[token] + 1
            else:
                vocaball[token] = 1

        vocabtokens = [key for key in vocaball if vocaball[key] >= min_freq]
        vocabtokens = sorted(vocabtokens)

        vocab = dict()
        col = 0
        for tuple in vocabtokens:
            vocab[tuple] = col
            col = col + 1

    indptr = [0]
    indices = []
    data = []
    for feats in featslist:
        for tuple in feats:
            if vocab.__contains__(tuple[0]) and tuple[1] > 0:
                indices.append(vocab[tuple[0]])
                data.append(tuple[1])
        indptr.append(len(indices))

    X = csr_matrix((data, indices, indptr), shape=(len (featslist), len(vocab)), dtype='int64')
    return X, vocab

    pass

def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(len(labels), k)
    accuracies = []
    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)

    return avg
    pass



def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    clf = LogisticRegression()
    results = []
    for min_freq in min_freqs:
        for Ln in range(1, len(feature_fns) + 1):
            for funset in combinations(feature_fns, Ln):
                docs_list = [tokenize(doc, False) for doc in docs]
                X, vocab = vectorize(docs_list, funset, min_freq)
                if vocab:
                    accuracy = cross_validation_accuracy(clf, X, labels, 5)
                    d = dict()
                    d['features'] = funset
                    d['punct'] = False
                    d['accuracy'] = accuracy
                    d['min_freq'] = min_freq
                    results.append(d)
    return sorted (results, key=lambda x: (-x['accuracy'], -x['min_freq']))
    pass

def fit_best_classifier(docs, labels, best_result):
    min_freq = best_result['min_freq']
    featfun = best_result['features']
    punct = best_result['punct']

    docs_list = [tokenize(doc, punct) for doc in docs]

    X, vocab = vectorize(docs_list, featfun, min_freq)

    clf = LogisticRegression()
    clf.fit(X, labels)

    return clf, vocab

    pass

def parse_test_data(best_result, vocab):

    test_docs = []

    file = open('tweets.txt', 'r')
    rows = file.read().split('\n')
    for row in rows:
        if row:
            tweet = json.loads(row)
            tweet = tweet.lower()
            tweet = re.sub('@\S+', ' ', tweet)
            tweet = re.sub('http\S+', ' ', tweet)
            tweet = re.sub('\W+', ' ', tweet)
            test_docs.append(tweet)

    min_freq = best_result['min_freq']
    featfun = best_result['features']
    punct = best_result['punct']

    docs_list = [tokenize(doc, punct) for doc in test_docs]

    X, vocab = vectorize(docs_list, featfun, min_freq, vocab)

    return test_docs, X

    pass

def get_predictions(clf, X_test, test_docs):

    predicted = clf.predict(X_test)
    probas = clf.predict_proba(X_test)
    pos_result = []
    neg_result = []
    i = 0
    for doc in test_docs:
        record = dict()
        record['predicted'] = predicted[i]
        record['probability'] = probas[i]
        record['doc'] = doc
        if predicted[i] == 0:
            neg_result.append(record)
        elif predicted[i] == 1:
            pos_result.append(record)
        i = i + 1

    pos_sortedresult = sorted(pos_result, key=lambda x: -x['probability'][x['predicted']])
    neg_sortedresult = sorted(neg_result, key=lambda x: -x['probability'][x['predicted']])

    return pos_sortedresult, neg_sortedresult
    pass

def generate_result(test_docs, pos_results, neg_results):
    result = dict()
    result['Number of messages collected : '] = len(test_docs)
    result['Number of positive instances : '] = len(pos_results)
    result['Number of negative instances : '] = len(neg_results)
    result['Example of positice instance : '] = pos_results[0]['doc']
    result['Example of negative instance : '] = neg_results[0]['doc']

    config = configparser.RawConfigParser()
    config.read('config.properties')
    movie_name = config.get('CollectionSection', 'movie.name')

    print('------------------------------------------------------')
    print (int((len(pos_results) / len(test_docs)) * 100), '% of people are positive about the movie', movie_name)
    print (int((len(neg_results) / len(test_docs)) * 100), '% of people are negative about the movie', movie_name)
    print('**The result totally depends on the data used for analysis and might be wrong for a small set of random tweets')
    print('------------------------------------------------------')

    result_file = open("result_classify.txt", "w")
    result_file.write(json.dumps(result))

    pass


def main():

    if os.path.exists("tweets.txt"):

        feature_fns = [token_features, token_pair_features, lexicon_features]
        #Reading training data
        docs, labels = read_data(os.path.join('data', 'train'))
        read_afinnwords ()

        #to get the best combination for the classifier
        print ('Finding the best combination for the classifier')
        results = eval_all_combinations(docs, labels, [True, False], feature_fns, [2, 5, 10])
        best_result = results[0]
        print ('Using the best combination to train the classifier')
        clf, vocab = fit_best_classifier(docs, labels, results[0])

        print ('Preparing the test docs')
        test_docs, X_test = parse_test_data(best_result, vocab)

        print ('Predicting the labels for the test docs')
        pos_results, neg_results = get_predictions(clf, X_test, test_docs)
        print ('Storing the result')
        generate_result(test_docs,pos_results, neg_results)
        print ('Finished!')
    else:
        print("No tweets were found")
        print("Please run collect.py first ")

if __name__ == '__main__':
    main()