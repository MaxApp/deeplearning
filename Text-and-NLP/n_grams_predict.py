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

def build_vocab(tokenized_sentences, threshold):
    """
    create a vocabulary with token frequences >= threshold
    """
    word_counts = count_words(tokenized_sentences)
    vocabulary = [word for word, cnt in word_counts.items() if cnt >= threshold]
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



