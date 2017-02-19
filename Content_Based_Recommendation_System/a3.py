# coding: utf-8

#
#
# I have implemented a content-based recommendation algorithm.
# It uses the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/


from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """

    movies['tokens'] = pd.Series([tokenize_string(gn) for gn in movies['genres']])
    return movies

    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO

    #df will be used for calculating tfidf
    df = dict()

    tokens_in_movies = movies['tokens']

    def unique_tokens(tokens):
        uni_tokens = set(tokens)
        return list(uni_tokens)

    # Creating df
    for tokens in tokens_in_movies:
        tokens = unique_tokens(tokens)
        for token in tokens:
            if df.__contains__(token):
                df[token] = df[token] + 1
            else:
                df[token] = 1

    vocab_tokens = [key for key in df]
    vocab_tokens = sorted(vocab_tokens)

    #Creating vocab from df
    vocab = dict()
    col = 0
    for term in vocab_tokens:
        vocab[term] = col
        col = col + 1


    def get_tf(token, tokens):
        count = 0
        for tk in tokens:
            if tk == token:
                count = count + 1
        return count

    def get_max_k(tokens):
        tokenCounts = dict()
        for token in tokens:
            if tokenCounts.__contains__(token):
                tokenCounts[token] = tokenCounts[token] + 1
            else:
                tokenCounts[token] = 1
        return sorted(tokenCounts.items(), key = lambda x:-x[1])[0][1]

    csr_list = []
    N = len(tokens_in_movies)
    indptr = [0]
    indices = []
    data = []
    for tokens in tokens_in_movies:
        for token in tokens:
            if vocab.__contains__(token):
                indices.append(vocab[token])
                idf = math.log10(N / df[token])
                val = float(float(get_tf(token, tokens) / get_max_k(tokens)) * idf)
                data.append(val)
        indptr.append(len(indices))
        csr_list.append(csr_matrix((data, indices, indptr), shape=(1, len(vocab))))
        indptr = [0]
        indices = []
        data = []

    movies['features'] = pd.Series(csr_list)
    return movies, vocab
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """

    k = list(a.toarray()[0])
    l = list(b.toarray()[0])
    return sum(x*y for x,y in zip(k,l)) / ((math.sqrt(sum([z*z for z in k]))) * (math.sqrt(sum([z*z for z in l]))))

    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """

    predicted_ratings = []
    for index, row in ratings_test.iterrows():
        ratings_list = []
        cosine_vals = []
        all_ratings = []
        found = False
        for index1, row1 in ratings_train[ratings_train.userId == row.userId].iterrows():
            all_ratings.append(row1.rating)
            cosine = cosine_sim(movies[movies.movieId==row.movieId]['features'].iloc[0], movies[movies.movieId==row1.movieId]['features'].iloc[0])
            if (cosine > 0):
                found = True
                cosine_vals.append(cosine)
                ratings_list.append(cosine * row1.rating)

        if not found:
            predicted_ratings.append(np.mean(all_ratings))
        else:
            predicted_ratings.append(float(sum(ratings_list) / sum(cosine_vals)))

    return np.array(predicted_ratings)
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
