import torch
import torch.nn as nn


class FusionLayer(nn.Module):

    def __init__(self,
                 bert_hidden_size=768,
                 demographic_size=64,
                 output_size=768,
                 dropout=0.2):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(
                bert_hidden_size + demographic_size,
                output_size
            ),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, bert_embedding, demographic_embedding):

        combined_features = torch.cat(
            (bert_embedding, demographic_embedding),
            dim=1
        )

        fused_embedding = self.fusion(combined_features)

        return fused_embedding