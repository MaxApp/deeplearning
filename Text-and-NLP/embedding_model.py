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
        {"ctx":["grape", "apple"], "center":"orange"},
        {"ctx":["car","bike"], "center":"plane"},
        {"ctx":["cat", "bird"], "center":"dog"},
        {"ctx":["plane","bike"], "center":"car"},
        {"ctx":["bird", "dog"], "center":"cat"},
        {"ctx":["orange", "grape"], "center":"apple"},
        {"ctx":["plane","car"], "center":"bike"},
        {"ctx":["cat", "dog"], "center":"bird"},
        {"ctx":["cat", "bird"], "center":"dog"},
        {"ctx":["plane","car"], "center":"bike"},
        {"ctx":["orange", "grape"], "center":"apple"},
        {"ctx":["bird", "dog"], "center":"cat"},
        {"ctx":["orange", "apple"], "center":"grape"},
        {"ctx":["plane","car"], "center":"bike"},
        {"ctx":["orange", "grape"], "center":"apple"},
        {"ctx":["bird", "dog"], "center":"cat"},
        {"ctx":["grape", "apple"], "center":"orange"},
        {"ctx":["car","bike"], "center":"plane"},
    ]

    embedding_dim = 3
    embedding_model = CbowEmbeddingModel(len(vocabulary), embedding_dim)

    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    num_epoch = 120
    embedding_model.train()
    for i in range(num_epoch):
        epoch_loss = 0.0
        for context_and_center in train_data:
            ctx = context_and_center["ctx"]
            center = context_and_center["center"]

            # print(f"ctx: {ctx}  center: {center}")
            # convert idx
            ctx_idx = [word_to_idx[ctx_word] for ctx_word in ctx]
            center_idx = [word_to_idx[center]]
            # print(f"ctx_idx: {ctx_idx}  center_idx: {center_idx}")

            optimizer.zero_grad()
            output, _ = embedding_model(torch.tensor(ctx_idx).unsqueeze(0))
            # print(f"output: {output}")
            loss = loss_function(output, torch.tensor(center_idx))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_avg_loss = epoch_loss / len(train_data)
        if (i+1) % 10 == 0:
            print(f"Epoch: {i+1}/{num_epoch}   Loss: {epoch_avg_loss}")

    

        