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
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.param = nn.Parameter(torch.ones(d_model))
        self.eps = eps 
        self.d_model = d_model

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        sum_of_squares = torch.sum(x.square(), dim = -1, keepdim=True)
        mean_of_squares = sum_of_squares / self.d_model
        rms = torch.sqrt(mean_of_squares + self.eps)
        result =  x / rms * self.param
        return result.to(in_dtype)
        

def SiLU(x):
    return x * torch.sigmoid(x)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        dff = int(d_model * (8 / 3))
        dff = (dff + 63) // 64 *64 # making it a multiple of 64
        self.layer1 = LinearLayer(d_model, dff, device = device, dtype = dtype)
        self.layer2 = LinearLayer(dff, d_model, device = device, dtype = dtype)
        self.layer3 = LinearLayer(d_model, dff, device = device, dtype = dtype)

    def forward(self, x):
        W1x = self.layer1(x)
        W3x = self.layer3(x)
        final = self.layer2(SiLU(W1x) * W3x)
        return final 
    
    class RotaryPositionalEmbedding(nn.Module):
        def __inti__(self, theta, d_k, max_seq_len, device = None):
            super().__init__()
            self.theta = theta 
            if theta != 0:
                pos = torch.arange(max_seq_len, deice = device)[:, None]
                dim = torch.arange(0, d_k, device = device)[None, :]
                inv_frq = 1.0 / (theta **(dim/d_k))
                freqs = pos * inv_frq

                self.register_buffer("cos", freqs.cos(), persistent = False)
                self.register_buffer("sin", freqs.sin(), persistent=False)          

        def forward(self, x, token_positions):
            if self.theta == 0:
                return x 
            
            seq_len = x.shape[-2]
            if token_positions is None:
                cos = self.cos[:seq_len]
                sin = self.sin[:seq_len]
            else:
                cos = self.cos[token_positions]
                sin = self.sin[token_positions]

            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

            x_even = x[..., ::2]
            x_odd = x[...,1::2]

            out_even = x_even*cos - x_odd*sin 
            out_odd = x_even*sin + x_odd*cos 

            out = torch.empty_like(x)
            out[..., ::2]= out_even 
            out[..., 1::2]=out_odd 

            return out 

if __name__ == "__main__":
    print("hello")
    model  = LinearLayer(2,3)
    data= torch.randn(12,2)
    output = model(data)
    print(output, "/n")

