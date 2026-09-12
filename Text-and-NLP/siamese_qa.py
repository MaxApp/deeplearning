import torch

def TripletLossFn(v1, v2, margin=0.25):
    """Custom Loss function.

    Args:
        v1 (torch.Tensor): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (torch.Tensor): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        torch.Tensor: Triplet Loss.
    """
    
    # Take the dot product of the two batches.
    scores = torch.matmul(v1, v2.transpose(0, 1))
    # calculate new batch size
    batch_size = len(scores)
    positive = torch.diagonal(scores)
    eye = torch.eye(batch_size, device=scores.device, dtype=scores.dtype)
    negative_without_positive = scores - eye * 2.0
    # take the row by row `max` of `negative_without_positive`. 
    # Hint: negative_without_positive.max(axis = [?])  
    closest_negative = negative_without_positive.max(dim=1).values
    # Subtract the diagonal from the negative scores before averaging.
    negative_zero_on_duplicate = (1.0 - eye) * scores
    mean_negative = torch.sum(negative_zero_on_duplicate, dim=1) / (batch_size - 1)
    # A = subtract `positive` from `margin` and add `closest_negative` 
    triplet_loss1 = torch.maximum(margin - positive + closest_negative, torch.zeros_like(positive))
    # B = subtract `positive` from `margin` and add `mean_negative`
    triplet_loss2 = torch.maximum(margin - positive + mean_negative, torch.zeros_like(positive))
    # Add the two losses together and take their mean.
    triplet_loss = torch.mean(triplet_loss1 + triplet_loss2)
    
    return triplet_loss