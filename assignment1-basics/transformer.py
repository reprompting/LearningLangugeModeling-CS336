import torch 
import torch.nn as nn 
import math 

class LinearLayer(nn.Module):
    def __init__(self, in_feature, out_feature, device=None, dtype=None):
        super().__init__()

        self.in_features = in_feature 
        self.out_features = out_feature 
        self.weight = nn.Parameter(torch.empty(out_feature, in_feature))
        std = math.sqrt( 2 / (in_feature + out_feature))
        nn.init.trunc_normal_(self.weight, std=std)

    def forward(self, x):
        return ((x@self.weight.T))
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.embed, a = -3, b=3)

    def forward(self, token_ids):
        return self.embed[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None)
        super().__init__()

    def forward(self, x):
        pass 


if __name__ == "__main__":
    print("hello")
    model  = LinearLayer(2,3)
    data= torch.randn(12,2)
    output = model(data)
    print(output, "/n")

