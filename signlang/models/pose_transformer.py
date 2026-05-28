from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from signlang.topology import FRAME_DIM


@dataclass
class PoseTransformerConfig:
    vocab_size: int
    frame_dim: int = FRAME_DIM
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    max_gloss_tokens: int = 64
    max_frames: int = 256
    pad_token_id: int = 0


class Vocabulary:
    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, token_to_id=None):
        self.token_to_id = token_to_id or {
            self.pad_token: 0,
            self.unk_token: 1,
        }
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @classmethod
    def build(cls, gloss_sequences):
        vocab = cls()
        for sequence in gloss_sequences:
            for token in sequence.strip().split():
                if token not in vocab.token_to_id:
                    index = len(vocab.token_to_id)
                    vocab.token_to_id[token] = index
                    vocab.id_to_token[index] = token
        return vocab

    @classmethod
    def from_dict(cls, data):
        return cls({token: int(index) for token, index in data["token_to_id"].items()})

    def to_dict(self):
        return {"token_to_id": self.token_to_id}

    def encode(self, sequence):
        tokens = sequence.strip().split()
        if not tokens:
            tokens = [self.unk_token]
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]

    def __len__(self):
        return len(self.token_to_id)


class PoseTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gloss_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_dim,
            padding_idx=config.pad_token_id,
        )
        self.frame_projection = nn.Linear(config.frame_dim, config.hidden_dim)
        self.source_position = SinusoidalPositionalEncoding(config.hidden_dim, config.max_gloss_tokens)
        self.target_position = SinusoidalPositionalEncoding(config.hidden_dim, config.max_frames)
        self.transformer = nn.Transformer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_dim, config.frame_dim + 1)

    def forward(self, gloss_tokens, decoder_frames, gloss_padding_mask=None, frame_padding_mask=None):
        source = self.source_position(self.gloss_embedding(gloss_tokens))
        target = self.target_position(self.frame_projection(decoder_frames))
        target_mask = _causal_mask(target.size(1), target.device)
        output = self.transformer(
            source,
            target,
            tgt_mask=target_mask,
            src_key_padding_mask=gloss_padding_mask,
            tgt_key_padding_mask=frame_padding_mask,
            memory_key_padding_mask=gloss_padding_mask,
        )
        raw = self.output(output)
        coords = torch.sigmoid(raw[..., : self.config.frame_dim])
        eos_logits = raw[..., self.config.frame_dim]
        return coords, eos_logits

    @torch.no_grad()
    def greedy_decode(self, gloss_tokens, max_frames=160, eos_threshold=0.6, min_frames=8):
        self.eval()
        device = next(self.parameters()).device
        gloss_tokens = gloss_tokens.to(device)
        if gloss_tokens.ndim == 1:
            gloss_tokens = gloss_tokens.unsqueeze(0)
        padding_mask = gloss_tokens.eq(self.config.pad_token_id)
        decoder_frames = torch.zeros(
            (gloss_tokens.size(0), 1, self.config.frame_dim),
            dtype=torch.float32,
            device=device,
        )
        generated = []
        for frame_index in range(max_frames):
            coords, eos_logits = self.forward(gloss_tokens, decoder_frames, padding_mask)
            next_frame = coords[:, -1, :]
            eos_score = torch.sigmoid(eos_logits[:, -1])
            generated.append(next_frame)
            decoder_frames = torch.cat([decoder_frames, next_frame.unsqueeze(1)], dim=1)
            if frame_index + 1 >= min_frames and torch.all(eos_score > eos_threshold):
                break
        return torch.stack(generated, dim=1)

    def checkpoint_payload(self, vocab):
        return {
            "model_state": self.state_dict(),
            "config": asdict(self.config),
            "vocab": vocab.to_dict(),
        }


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        encoding = torch.zeros(max_length, hidden_dim)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values):
        return values + self.encoding[:, : values.size(1), :]


def _causal_mask(size, device):
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

