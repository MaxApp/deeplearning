import torch
from torch import nn
from torch.nn import functional as F

def TripletLossFn(v1, v2, margin=0.25):
    """Custom Loss function.

    Args:
        v1 (torch.Tensor): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (torch.Tensor): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        torch.Tensor: Triplet Loss.
    """
    
    scores = torch.matmul(v1, v2.transpose(0, 1))
    # calculate new batch size
    batch_size = len(scores)
    positive = torch.diagonal(scores)
    eye = torch.eye(batch_size, device=scores.device, dtype=scores.dtype)
    negative_without_positive = scores - eye * 2.0
    # take the row by row `max` of `negative_without_positive`. 
    closest_negative = negative_without_positive.max(dim=1).values
    # subtract the diagonal from the negative scores before averaging.
    negative_zero_on_duplicate = (1.0 - eye) * scores
    mean_negative = torch.sum(negative_zero_on_duplicate, dim=1) / (batch_size - 1)
    # A = subtract `positive` from `margin` and add `closest_negative` 
    triplet_loss1 = torch.maximum(margin - positive + closest_negative, torch.zeros_like(positive))
    # B = subtract `positive` from `margin` and add `mean_negative`
    triplet_loss2 = torch.maximum(margin - positive + mean_negative, torch.zeros_like(positive))
    # add the two losses together and take their mean.
    triplet_loss = torch.mean(triplet_loss1 + triplet_loss2)
    
    return triplet_loss


class SiameseModel(nn.Module):
    """Encode two token sequences with the same embedding and LSTM."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)

    def encode(self, tokens):
        """Return one normalized representation per token sequence."""
        embeddings = self.embedding(tokens)
        outputs, _ = self.lstm(embeddings)
        pooled = outputs.mean(dim=1)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, q1, q2):
        return self.encode(q1), self.encode(q2)


def Siamese(vocab_size, d_model=128, mode='train'):
    """Return a PyTorch Siamese model.

    Args:
        vocab_size (int): Length of the vocabulary.
        d_model (int, optional): Depth of the model. Defaults to 128.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to 'train'.

    Returns:
        SiameseModel: A Siamese model.
    """

    if mode not in {'train', 'eval', 'predict'}:
        raise ValueError("mode must be 'train', 'eval', or 'predict'")

    model = SiameseModel(vocab_size=vocab_size, d_model=d_model)
    if mode in {'eval', 'predict'}:
        model.eval()
    return model