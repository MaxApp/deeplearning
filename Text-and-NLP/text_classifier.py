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
        pass

    def __len__(self):
        """Returns the vocabulary size."""
        return len(self.word2idx)