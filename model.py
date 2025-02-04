import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2,)
        # apply sin on even positions and cosine on odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # apply the batch dimension
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # If a tensor is registered as a buffer, it is saved in the state_dict of the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # the positional encoding is not trainable
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplcative param
        self.bias = nn.Parameter(torch.zeros(1)) # Additive param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 matrix
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(self.linear_1(x)))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.head_dim = d_model // h

        assert self.head_dim * h == d_model, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq matrix
        self.w_k = nn.Linear(d_model, d_model) # Wk matrix
        self.w_v = nn.Linear(d_model, d_model) # Wv matrix

        self.w_o = nn.Linear(d_model, d_model) # Wo matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def SelfAttention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, seq_len)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = scores @ value
        return output, scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = self.SelfAttention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        x = self.w_o(x) # (batch_size, seq_len, d_model)
        
        return x, self.attention_scores

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    
    def __init__(self, 
                 self_attn_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, 
                 normalization_layer: LayerNormalization,
                 dropout: float
                ) -> None:
        super().__init__()

        self.self_attn_block = self_attn_block
        self.feed_forward_block = feed_forward_block
        self.normalization_layer = normalization_layer
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)])
