import random
import pandas as pd
import numpy as np

# Preprocessing =================================================
# dataset from Kaggle: https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus

# combine each word('Word') of a sentence('Sentence #') to a list
# also with corresponding labels('Tag')
def grab_sentences_labels(csv_file_path):
    # load csv dataset from disk
    df = pd.read_csv(csv_file_path, encoding = "ISO-8859-1")
    # print(df.head())

    # forward fill the 'Sentence #' columns in order to group by it
    df["Sentence #"] = df["Sentence #"].ffill()
    sentences = []
    labels = []

    for _, group in df.groupby("Sentence #", sort=False):
        sentences.append(group["Word"].tolist())
        labels.append(group["Tag"].tolist())

    return df, sentences, labels

def build_vocab(sentences, pad=0, unk=1):
    word2idx = {}
    idx2word = {}
    for sent in sentences:
        for w in sent:
            if w not in word2idx:
                idx2word[len(word2idx)] = w
                word2idx[w] = len(word2idx)

    word2idx.update({"<unk>": unk, "<pad>": pad})
    idx2word.update({unk: "<unk>", pad: "<pad>"})
    return word2idx, idx2word

def build_tag(labels):
    tag2idx = {}
    idx2tag = {}

    for labseq in labels:
        for tag in labseq:
            if tag not in tag2idx:
                idx2tag[len(tag2idx)] = tag
                tag2idx[tag] = len(tag2idx)

    return tag2idx, idx2tag

def tokenize_sent_labels(sentences, labels, word2idx, tag2idx):
    X = []
    Y = []
    for sent, lab in zip(sentences, labels):
        x = [word2idx.get(tok, word2idx["<unk>"]) for tok in sent]
        y = [tag2idx[tag] for tag in lab]
        X.append(x)
        Y.append(y)
    return X, Y


# self define dataloader with batches =================================================
def data_loader(batch_size, x, y, pad, shuffle=False):
    '''
    Self defined dataloader with padding, implemented as a generator.
    '''
    
    # total sentences
    num_lines = len(x)
    
    # indicies of sentences that can be shuffled
    lines_index = [*range(num_lines)]
    if shuffle:
        random.shuffle(lines_index)

    # index of indicies which maybe shuffled
    index = 0
    while True:
        buffer_x = []
        buffer_y = []
        
        # copy from x[index : index + batch_size] 
        # along with corresponding labels y[index : index + batch_size]

        max_len = 0 # the max_len of sentence in this batch
        for i in range(batch_size):
            if index >= num_lines:
                # if reach the end of dataset, then reset the index to 0
                index = 0
                if shuffle:
                    # shuffe indicies for each batch
                    random.shuffle(lines_index)

            sent = x[lines_index[index]]
            buffer_x.append(sent)            
            buffer_y.append(y[lines_index[index]])

            # record the max len in this batch
            cur_len = len(sent)
            if cur_len > max_len:
                max_len = cur_len
            
            index += 1

        # (batch_size, max_len) 'full' of pad
        X = np.full((batch_size, max_len), pad)
        Y = np.full((batch_size, max_len), pad)

        # copy from lists to np arrays
        for i in range(batch_size):
            X[i,:len(x[i])] = x[i]
            Y[i,:len(y[i])] = y[i]

        yield((X,Y))


# def NamedEntityRecognitionModel(vocab_size=35181, d_model=50, tags=tag_map):
    '''
      Input: 
        vocab_size - integer containing the size of the vocabulary
        d_model - integer describing the embedding size
      Output:
        model - a trax serial model
    '''
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    # model = tl.Serial(
    #   tl.Embedding(vocab_size=vocab_size, d_feature=d_model), # Embedding layer
    #   tl.LSTM(n_units=d_model), # LSTM layer
    #   tl.Dense(n_units=len(tags)), # Dense layer with len(tags) units
    #   tl.LogSoftmax()  # LogSoftmax layer
    #   )
    #   ### END CODE HERE ###
    # return model




if __name__ == "__main__":

    # load csv dataset from disk
    csv_file_path = r"E:\PDF\NLP\C3W2\ner_dataset_small.csv"
    _, sentences, labels = grab_sentences_labels(csv_file_path)

    # print(f"sentences: {len(sentences)},  labels: {len(labels)}")

    pad_num = 35180
    unk_num = 35179
    
    batch_size = 5
    mini_sentences = sentences[0: 8]
    mini_labels = labels[0: 8]

    tag2idx, idx2tag = build_tag(labels=labels)
    w2i, i2w = build_vocab(sentences=sentences, pad=pad_num, unk=unk_num)
    # print(w2i)

    x_8, y_8 = tokenize_sent_labels(mini_sentences, mini_labels, word2idx=w2i, tag2idx=tag2idx)
    dg = data_loader(batch_size, x_8, y_8, 35180, shuffle=False)
    X1, Y1 = next(dg)
    X2, Y2 = next(dg)
    print(Y1.shape, X1.shape, Y2.shape, X2.shape)
    print(X1[0][:], "\n", Y1[0][:])

    

    # import pprint
    # print(tag2idx)
    # print(idx2tag)
    # print(w2i)
    # print(w2i)
