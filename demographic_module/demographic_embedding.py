import torch
import torch.nn as nn


class DemographicEmbedding(nn.Module):

    def __init__(self,num_genders,num_races,num_religions,embedding_dim=32):
        super().__init__()
        self.gender_embedding = nn.Embedding(            num_embeddings=num_genders,
        embedding_dim=embedding_dim)

        self.race_embedding = nn.Embedding(            num_embeddings=num_races,
        embedding_dim=embedding_dim)

        self.religion_embedding = nn.Embedding(     num_embeddings=num_religions,
        embedding_dim=embedding_dim)

    def forward(self,gender_ids,race_ids,religion_ids):
        gender_vector = self.gender_embedding(gender_ids)
        race_vector = self.race_embedding(race_ids)
        religion_vector = self.religion_embedding(religion_ids)
        demographic_vector = torch.cat((gender_vector,                race_vector,religion_vector),dim=1)
        
        return demographic_vector