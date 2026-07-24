import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import utils

# define a simple model
class MyEmbeddingModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # linear layer
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded_vector = self.embedding(context)
        # Note: you could input several words with context window,
        # in that case, you would using average or sum of context tensor as CBOW
        # embedded_avg = torch.mean(embedded_vector, dim=1)

        # for simplicity, we just use one-to-one training pair
        output = self.linear(embedded_vector)
        return output, embedded_vector

def word_idx_dict(vocabulary: list):
    word_to_idx = {word:i for i, word in enumerate(vocabulary)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    return word_to_idx, idx_to_word


if __name__ == "__main__":
    # build a simple vocabulary
    vocabulary = ["car", "bike", "plane", "boat",
                  "cat", "dog", "bird", "horse",
                  "orange", "apple", "grape", "banana"]
    
    categories = {
        'Vehicles': ['car', 'bike', 'plane', 'boat'],
        'Pets': ['cat', 'dog', 'bird', 'horse'],
        'Fruits': ['orange', 'apple', 'grape', 'banana']
    }

    # word and indicies mappings
    word_to_idx, idx_to_word = word_idx_dict(vocabulary)

    # for a small dataset and simplify the code, 
    # construct the context and center word pairs manually
    """
    train_data = [
        {"ctx": ["bike", "plane", "boat"], "center": "car"},
        {"ctx": ["car", "bike", "boat"], "center": "plane"},
        {"ctx": ["dog", "bird", "horse"], "center": "cat"},
        {"ctx": ["cat", "bird", "horse"], "center": "dog"},
        {"ctx": ["cat", "dog", "horse"], "center": "bird"},
        {"ctx": ["cat", "dog", "bird"], "center": "horse"},
        # {"ctx": ["dog", "bird", "horse"], "center": "banana"}, 
        # {"ctx": ["cat", "dog", "bird"], "center": "apple"},
        {"ctx": ["apple", "grape", "banana"], "center": "orange"},
        # {"ctx": ["plane", "boat", "car"], "center": "grape"},
        {"ctx": ["car", "plane", "boat"], "center": "bike"},
        {"ctx": ["orange", "grape", "banana"], "center": "apple"},
        {"ctx": ["car", "bike", "plane"], "center": "boat"},
        {"ctx": ["orange", "apple", "banana"], "center": "grape"},
        # {"ctx": ["car", "bike", "boat"], "center": "dog"},
        {"ctx": ["orange", "apple", "grape"], "center": "banana"},
        # {"ctx": ["car", "bike", "plane"], "center": "cat"},
        # {"ctx": ["orange", "apple", "grape"], "center": "boat"},
    ]
    """

    train_data = [
        ('car', 'bike'),
        ('car', 'plane'),
        ('car', 'boat'),
        ('bike', 'car'),
        ('bike', 'plane'),
        ('bike', 'boat'),
        ('plane', 'car'),
        ('plane', 'bike'),
        ('plane', 'boat'),
        ('boat', 'car'),
        ('boat', 'bike'),
        ('boat', 'plane'),
        ('cat', 'dog'),
        ('cat', 'bird'),
        ('cat', 'horse'),
        ('dog', 'cat'),
        ('dog', 'bird'),
        ('dog', 'horse'),
        ('bird', 'cat'),
        ('bird', 'dog'),
        ('bird', 'horse'),
        ('horse', 'cat'),
        ('horse', 'dog'),
        ('horse', 'bird'),
        ('orange', 'apple'),
        ('orange', 'grape'),
        ('orange', 'banana'),
        ('apple', 'orange'),
        ('apple', 'grape'),
        ('apple', 'banana'),
        ('grape', 'orange'),
        ('grape', 'apple'),
        ('grape', 'banana'),
        ('banana', 'orange'),
        ('banana', 'apple'),
        ('banana', 'grape')
    ]


    embedding_dim = 5
    embedding_model = MyEmbeddingModel(len(vocabulary), embedding_dim)

    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    num_epoch = 300
    embedding_model.train()
    for i in range(num_epoch):
        epoch_loss = 0.0
        for ctx, center in train_data:
            # ctx = context_and_center["ctx"]
            # center = context_and_center["center"]

            # print(f"ctx: {ctx}  center: {center}")
            # convert idx
            # ctx_idx = [word_to_idx[ctx_word] for ctx_word in ctx]
            ctx_idx = word_to_idx[ctx]
            center_idx = word_to_idx[center]
            # print(f"ctx_idx: {ctx_idx}  center_idx: {center_idx}")

            optimizer.zero_grad()
            output, _ = embedding_model(torch.tensor(ctx_idx).unsqueeze(0))
            # print(f"output: {output}")
            loss = loss_function(output, torch.tensor(center_idx).unsqueeze(0))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_avg_loss = epoch_loss / len(train_data)
        if (i+1) % 10 == 0:
            print(f"Epoch: {i+1}/{num_epoch}   Loss: {epoch_avg_loss}")

    # fetch embeddings after training
    embedding_model.eval()
    all_embeddings = embedding_model.embedding.weight.detach().numpy()

    # use PCA to visualize performance
    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(all_embeddings)

    utils.plot_embeddings(coords=coords, labels=vocabulary, label_dict=categories, title='Word Embeddings PCA view')

        