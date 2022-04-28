import numpy as np
import torch
import torch.nn as nn
from .initialization import linear_init
from .token import Token2Embedding, tokenize, bos_token


class SLATE(nn.Module):
    def __init__(self, resolution, patch_ae, slot_attn, gpt_decoder,
                 use_memory_mask=False):
        super().__init__()

        self.resolution = resolution
        self.patch_ae = patch_ae
        self.gpt_decoder = gpt_decoder
        self.slot_attn = slot_attn
        self.use_memory_mask = use_memory_mask

        vocab_size = patch_ae.latent_size
        slot_size = slot_attn.slot_size
        transformer_dim = gpt_decoder.d_model

        # +1 for BOS token, which needs its own embedding
        self.token2emb = Token2Embedding(vocab_size + 1, transformer_dim,
                                         np.prod(resolution) + 1)

        # Transformer input
        transformer_dim = gpt_decoder.d_model
        self.slot_proj = nn.Linear(slot_size, transformer_dim, bias=False)
        linear_init(self.slot_proj, activation=None)

        # Transformer output
        self.emb2token = nn.Linear(transformer_dim, vocab_size, bias=False)
        linear_init(self.emb2token, activation=None)

    def forward(self, inputs):
        recons, (z, logits) = self.patch_ae(inputs)

        tokens = tokenize(z.detach().flatten(1, 2), add_bos=True)
        all_embeddings = self.token2emb(tokens)

        # no BOS token in SlotAttention input, right-shift Transformer input
        slot_input, tf_input = all_embeddings[:, 1:], all_embeddings[:, :-1]

        slots, attn_weights = self.slot_attn(slot_input)

        # Compute autoregressive prior using TransformerDecoder
        proj = self.slot_proj(slots)
        mask = attn_weights.detach() if self.use_memory_mask else None

        token_logits = self.emb2token(self.gpt_decoder(tf_input, proj, mask))

        return (recons, (slots, attn_weights), (z, logits),
                (token_logits, tokens[:, 1:, 1:]))  # exclude BOS from targets

    def embed(self, inputs):
        z = self.patch_ae.embed(inputs)

        tokens = tokenize(z.flatten(1, 2), add_bos=False)
        embedding = self.token2emb(tokens)

        slots, attn_weights = self.slot_attn(embedding)

        return slots, attn_weights

    def decoder(self, slots):
        sampled_tokens = self.sample_autoregressive(*slots)
        sampled_tokens = sampled_tokens.unflatten(1, self.resolution)
        return self.patch_ae.decoder(sampled_tokens.to(dtype=torch.float32))

    def sample_autoregressive(self, slots, attn_weights=None):
        slot_proj = self.slot_proj(slots)

        if attn_weights is not None and self.use_memory_mask:
            mask = attn_weights.detach()
        else:
            mask = None

        new_token = bos_token(slots, self.token2emb.vocab_size)
        sampled_emb, sampled_tokens = self.token2emb(new_token), []

        for i in range(np.prod(self.resolution)):
            if i > 0:
                new_e = self.token2emb(new_token, start_pos=i)
                sampled_emb = torch.cat([sampled_emb, new_e], dim=1)

            u = self.gpt_decoder(sampled_emb, slot_proj, mask)[:, -1:]

            new_token = tokenize(self.emb2token(u), add_bos=False)
            sampled_tokens.append(new_token[..., 1:])

        return torch.cat(sampled_tokens, dim=1)
