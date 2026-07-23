import torch
import torch.nn as nn
import utils

# define a training model
class CbowEmbeddingModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # linear layer
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded_vector = self.embedding(context)
        # average context
        embedded_avg = torch.mean(embedded_vector, dim=1)
        output = self.linear(embedded_avg)
        return output, embedded_vector

def word_idx_dict(vocabulary: list):
    word_to_idx = {word:i for i, word in enumerate(vocabulary)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    return word_to_idx, idx_to_word

if __name__ == "__main__":
    # build a simple vocabulary
    vocabulary = ["car", "bike", "plane", 
                  "cat", "dog", "bird", 
                  "orange", "apple", "grape"]
    # word and indicies mappings
    word_to_idx, idx_to_word = word_idx_dict(vocabulary)

    # for a small dataset and simplify the code, 
    # construct the context and center word pairs manually
    train_data = [
        {"ctx":["car","bike"], "center":"plane"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["plane","bike"], "center":"car"}
    ]
                  

    embedding_dim = 5
    
    embedding_model = CbowEmbeddingModel(len(vocabulary), embedding_dim)