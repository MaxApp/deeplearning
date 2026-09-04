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

def count_words(tokenized_sentences) -> dict:
    """
    count the number of appearence with each token
    """
    word_counts = defaultdict(int)
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] += 1
    
    return dict(word_counts)

def build_vocab(tokenized_sentences, threshold, end_token='</s>', unknown_token='<unk>'):
    """
    create a vocabulary with token frequences >= threshold
    """
    vocabulary = [end_token, unknown_token]
    word_counts = count_words(tokenized_sentences)
    vocabulary += [word for word, cnt in word_counts.items() if cnt >= threshold]

    return vocabulary

def replace_by_unk(tk_sentences, vocabulary, unknown_token="<unk>"):
    """
    replace unknown tokens with '<unk>'
    """
    for sentence in tk_sentences:
        for i in range(len(sentence)):
            if sentence[i] not in vocabulary:
                sentence[i] = unknown_token
    return tk_sentences

def preprocess_data(vocabulary, train_data, test_data):
    train_data_replaced = replace_by_unk(train_data, vocabulary)
    test_data_replaced = replace_by_unk(test_data, vocabulary)    
    return train_data_replaced, test_data_replaced


# ------ N-grams functions ------------

def n_grams_count(sentences, n, start_token='<s>', end_token = '</s>') -> dict:    

    n_grams = defaultdict(int)

    for sentence in sentences:
        # prepend <s> n-1 times, append </s> at the end
        sentence = [start_token] * (n-1) + sentence + [end_token]
        
        for i in range(len(sentence) - n + 1):
            n_gram = tuple(sentence[i:i+n])
            n_grams[n_gram] += 1
    
    return dict(n_grams)


def cal_smoothing_probability(word,
                              n_minus_one_gram,
                              n_minus_one_gram_counts,
                              n_gram_counts,
                              vocabulary_size, 
                              k=1.0):
    # convert list to tuple to use it as a dictionary key
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