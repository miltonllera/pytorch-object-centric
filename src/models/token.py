import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial import PositionEmbedding1D


def bos_token(inputs, vocab_size):
    bos = inputs.new_zeros(len(inputs), 1, vocab_size)
    bos[:, 0, 0] = 1
    return bos


def tokenize(token_probs, add_bos=True):
    index = token_probs.argmax(dim=-1)

    tokens = F.one_hot(index, token_probs.shape[-1]).to(dtype=torch.float32)
    tokens = F.pad(tokens, (1, 0), "constant", 0)

    if add_bos: # add begening-of-sequence token coded as [1, 0, ..., 0]
        tokens = torch.cat([bos_token(tokens, tokens.size(-1)), tokens], dim=1)

    return tokens


class Token2Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super().__init__()
        self.embedding_dict = nn.Embedding(vocab_size, embedding_dim)
        self.add_pos_emb = PositionEmbedding1D(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    @property
    def vocab_size(self):
        return self.embedding_dict.num_embeddings

    @property
    def embedding_dim(self):
        return self.embedding_dict.embedding_dim

    def forward(self, index_weights, start_pos=0):
        index = index_weights.argmax(dim=-1)
        embeddings = self.embedding_dict(index)
        return self.dropout(self.add_pos_emb(embeddings, start_pos))

    def reset_parameters(self):
        self.embedding_dict.reset_parameters()
        self.add_pos_emb.reset_parameters()
