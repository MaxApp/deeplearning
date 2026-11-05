"""A two-layer network with a manually implemented inverted-dropout"""

import torch


def inverted_dropout(activations: torch.Tensor, keep_probability: float = 0.8, training: bool = True):
    """Apply a dropout filter and scale during training"""
    if not 0 < keep_probability <= 1:
        raise ValueError("keep_probability must be greater than 0 and at most 1")

    if not training or keep_probability == 1:
        return activations

    dropout_filter = (
        torch.rand_like(activations) < keep_probability
    ).to(dtype=activations.dtype)

    # scale the activations to keep the expectation
    return activations * dropout_filter / keep_probability


class SimpleNetwork(torch.nn.Module):

    def __init__(self, input_features: int = 2, keep_probability: float = 0.8):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_features, 3)
        self.layer2 = torch.nn.Linear(3, 1)
        self.keep_probability = keep_probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = torch.relu(self.layer1(inputs))
        activations = inverted_dropout(
            activations,
            keep_probability=self.keep_probability,
            training=self.training
        )
        return self.layer2(activations)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = SimpleNetwork()
    inputs = torch.tensor([[0.5, 1.0], [1.5, -0.5]], dtype=torch.float32)

    model.train()
    print("Training output:", model(inputs))

    model.eval()
    with torch.no_grad():
        print("Evaluation output:", model(inputs))