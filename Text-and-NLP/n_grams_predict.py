import math
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')

def split_to_sentences(data):
    """
    split sentence by "\n"
    """
    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences

def tokenize_sentences(sentences):

    tokenized_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()
        # tokenize with `nltk` library
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)

    return tokenized_sentences

def split_dataset(tokenized_sentences):
    """
    train data set: 0.8
    test data set: 0.2
    """
    random.seed(87)
    random.shuffle(tokenized_sentences)

    train_size = int(len(tokenized_sentences) * 0.8)
    train_data = tokenized_sentences[0:train_size]
    test_data = tokenized_sentences[train_size:]
    return train_data, test_data

# --- create vocabulary -------
def build_vocab(tokenized_sentences, threshold, end_token='</s>', unknown_token='<unk>'):
    """
    create a vocabulary with token frequences >= threshold
    """
    # just include `<s>` and `<unk>`
    # do not include `<s>`, it will never be next word.
    vocabulary = [end_token, unknown_token]
    filtered_words = filter_words_by_theshold(tokenized_sentences, threshold=threshold)
    vocabulary += filtered_words
    return vocabulary

def filter_words_by_theshold(tokenized_sentences, threshold) -> list:
    """
    filter vocabulary words with threshold
    """
    word_counts = defaultdict(int)
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] += 1
    filtered_words = [word for word, cnt in word_counts.items() if cnt >= threshold]
    return filtered_words

# --- pre-process input text data -----
def preprocess_data(vocabulary, train_data, test_data, n_gram, start_token='<s>', end_token = '</s>', unknown_token='<unk>'):
    """
    1. replace tokens not in vocabulary with <unk>
    2. add <s> at the start of sentence and </s> at the end of sentence
    """
    train_data_replaced = replace_and_add(train_data, vocabulary, n_gram, start_token, end_token, unknown_token)
    test_data_replaced = replace_and_add(test_data, vocabulary, n_gram, start_token, end_token, unknown_token)    
    return train_data_replaced, test_data_replaced

def replace_and_add(tokenized_sentences, vocabulary, n, start_token, end_token, unknown_token):
    
    processed_sentences = []
    for sentence in tokenized_sentences:
        pro_sent = []
        for token in sentence:
            if token in vocabulary:
                pro_sent.append(token)
            else:
                pro_sent.append(unknown_token)

        # add start and end tokens
        pro_sent = [start_token] * (n - 1) + pro_sent + [end_token]
        processed_sentences.append(pro_sent)

    return processed_sentences


# ------ N-grams functions ------------

def n_grams_count(sentences, n, ) -> dict:
    """
    calculate N-grams count by given sentences
    return dictionary with `n-gram` as key, `count` as value
    """
    n_grams = defaultdict(int)
    for sentence in sentences:
        for i in range(len(sentence) - n + 1):
            n_gram = tuple(sentence[i:i+n])
            n_grams[n_gram] += 1
    return dict(n_grams)


def cal_smoothing_probability(word,
                              n_minus_one_gram, # words before next one
                              n_minus_one_gram_counts,
                              n_gram_counts,
                              vocabulary_size, 
                              k=1.0):
    # convert to tuple as dictionary key
    n_minus_one_gram = tuple(n_minus_one_gram)
    previous_n_minus_one_gram_count = n_minus_one_gram_counts.get(n_minus_one_gram, 0)    
    # k-smoothing
    denominator = previous_n_minus_one_gram_count + (k * vocabulary_size)

    n_gram = n_minus_one_gram + (word, )
    n_gram_count = n_gram_counts.get(n_gram, 0)
    # apply smoothing
    numerator = n_gram_count + k

    probability = numerator / denominator    
    return probability

def cal_all_probabilities(n_minus_one_gram, n_minus_one_gram_counts, n_gram_counts, vocabulary, k=1.0):
    """
    calculate the probabilities of next words using the n-gram counts with k-smoothing
    """
    vocabulary_size = len(vocabulary)
    
    probabilities = {}
    for word in vocabulary:
        probability = cal_smoothing_probability(word, n_minus_one_gram, 
                                                n_minus_one_gram_counts, n_gram_counts, 
                                                vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities

def make_count_matrix(n_gram_counts, vocabulary):
    """Display probability matrix by n-grams"""
    
    n_grams = []
    for n_gram in n_gram_counts.keys():
        n_gram = n_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    
    # mapping n-gram to row
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}
    # mapping next word to column
    col_index = {word:j for j, word in enumerate(vocabulary)}
    
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_gram, count in n_gram_counts.items():
        n_gram = n_gram[0:-1]
        word = n_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix