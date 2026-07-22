import torch
import torch.nn as nn


class TrustEmbedding(nn.Module):

    def __init__(self, embedding_dim=64):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU()
        )

    def forward(
        self,
        confidence,
        p_normal,
        p_offensive,
        p_hate
    ):

        trust_features = torch.stack(
            (
                confidence,
                p_normal,
                p_offensive,
                p_hate
            ),
            dim=1
        )

        trust_embedding = self.embedding(
            demographic_features.float()
        )

        return trust_embedding