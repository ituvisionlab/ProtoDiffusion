import torch
from torch import nn

class ConditionalEmbedding(nn.Module):
    """
    A conditional embedding module for incorporating label information into a model.

    Args:
        num_labels (int): Number of unique labels.
        d_model (int): Dimensionality of the model's embedding space.
        dim (int): Output dimensionality of the conditional embedding.

    Attributes:
        condEmbedding (nn.Sequential): Sequential layers for conditional embedding.
    """
    def __init__(self, num_labels:int, d_model:int, dim:int):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels, embedding_dim=d_model, padding_idx=0), 
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(t)
        return emb
