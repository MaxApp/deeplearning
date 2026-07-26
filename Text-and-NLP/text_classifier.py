import torch
from torch.utils.data import Dataset
from collections import Counter

class Vocabulary:
    """word-to-index management for vocabulary
       min_freq: threshold for word-frequency
    """
    def __init__(self, min_freq=1):
        # initial mappings for word-to-index and index-to-word
        # including '<pad>','<unk>'
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.min_freq = min_freq

    def build_vocab(self, texts):
        # the frequency of all words in the corpus
        word_counts = Counter(word for text in texts for word in text)
        
        # add words if they meet the minimum frequency
        for word, count in word_counts.items():
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text):
        """converts a tokenized text to a sequence of indices.
        """
        # Use the <unk> token for words not in the vocabulary.
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in text]

    def decode(self, indices):
        """convert indices to tokens"""
        return [self.idx2word.get(idx) for idx in indices]

    def __len__(self):
        """Returns the vocabulary size."""
        return len(self.word2idx)

class TextDataset(Dataset):
    """
    A custom PyTorch Dataset for handling text and label data.

    This class encapsulates a dataset of texts and their corresponding labels,
    making it compatible with PyTorch's DataLoader.
    """
    def __init__(self, texts, labels):
        """
        Initializes the TextDataset object.

        Args:
            texts: A list or array of numericalized text sequences.
            labels: A list or array of corresponding labels.
        """
        # Store the collection of texts.
        self.texts = texts
        # Store the collection of labels.
        self.labels = labels
        # Find unique class labels and store them
        self.classes = sorted(list(set(labels)))

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        # Return the size of the dataset based on the number of texts.
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at a given index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing the text and label as PyTorch tensors.
        """
        # Create a dictionary for the sample at the specified index.
        sample = {
            'text': torch.tensor(self.texts[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        # Return the sample dictionary.
        return sample