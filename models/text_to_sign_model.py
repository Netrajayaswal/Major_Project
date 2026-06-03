#!/usr/bin/env python3
"""
Text-to-Sign Translation Model Architecture
5-Stage Pipeline: Text → Gloss → Pose → Refinement → Output
Optimized for 4GB VRAM + 16GB RAM hardware constraints
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# =============================================================================
# STAGE 1: TEXT ENCODER
# =============================================================================

class TextEncoder(nn.Module):
    """
    Stage 1: Text Encoding using lightweight transformer.
    Uses DistilBERT-style architecture for memory efficiency.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_length: int = 128,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len] token indices
            attention_mask: [batch_size, seq_len] attention mask
        
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim]
            pooled_output: [batch_size, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(position_ids)
        
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.layer_norm(hidden_states)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True = ignore)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Apply transformer encoder
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=src_key_padding_mask)
        
        # Pool output (mean pooling over non-masked tokens)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled_output = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        return hidden_states, pooled_output


# =============================================================================
# STAGE 2: TEXT-TO-GLOSS TRANSLATOR
# =============================================================================

class TextToGlossTranslator(nn.Module):
    """
    Stage 2: Seq2Seq Transformer for Text → Gloss translation.
    Sign languages have different grammar, so we need to translate.
    """
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        gloss_vocab_size: int = 1000,
        hidden_dim: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.gloss_vocab_size = gloss_vocab_size
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        
        # Projection to match dimensions
        self.text_to_hidden = nn.Linear(text_encoder.hidden_dim, hidden_dim)
        
        # Gloss embedding
        self.gloss_embedding = nn.Embedding(gloss_vocab_size, hidden_dim, padding_idx=pad_token_id)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, gloss_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        gloss_ids: Optional[Tensor] = None,
        max_length: int = 20,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: [batch_size, text_seq_len]
            attention_mask: [batch_size, text_seq_len]
            gloss_ids: [batch_size, gloss_seq_len] (for teacher forcing during training)
            max_length: maximum gloss sequence length (for inference)
        
        Returns:
            Dictionary with logits and predicted gloss sequences
        """
        # Encode text
        text_hidden, text_pooled = self.text_encoder(input_ids, attention_mask)
        text_hidden = self.text_to_hidden(text_hidden)  # [batch, text_len, hidden_dim]
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if gloss_ids is not None:
            # Training mode: use teacher forcing
            return self._forward_train(text_hidden, attention_mask, gloss_ids)
        else:
            # Inference mode: autoregressive generation
            return self._forward_infer(text_hidden, attention_mask, max_length, device)
    
    def _forward_train(
        self,
        text_hidden: Tensor,
        attention_mask: Optional[Tensor],
        gloss_ids: Tensor,
    ) -> Dict[str, Tensor]:
        """Training forward pass with teacher forcing."""
        # Create causal mask for decoder
        gloss_len = gloss_ids.shape[1]
        causal_mask = self._generate_causal_mask(gloss_len, text_hidden.device)
        
        # Embed gloss tokens (shift right for decoder input)
        decoder_input = gloss_ids[:, :-1]  # Remove last token
        gloss_embeds = self.gloss_embedding(decoder_input)
        
        # Create memory mask from text attention mask
        memory_mask = None
        if attention_mask is not None:
            memory_mask = (attention_mask == 0)  # True = ignore
        
        # Decode
        decoder_output = self.decoder(
            gloss_embeds,
            text_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return {
            "logits": logits,
            "hidden_states": decoder_output,
        }
    
    def _forward_infer(
        self,
        text_hidden: Tensor,
        attention_mask: Optional[Tensor],
        max_length: int,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        """Inference forward pass with autoregressive generation."""
        batch_size = text_hidden.shape[0]
        
        # Start with SOS token
        generated = torch.full(
            (batch_size, 1),
            self.sos_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Create memory mask
        memory_mask = None
        if attention_mask is not None:
            memory_mask = (attention_mask == 0)
        
        # Generate tokens
        for _ in range(max_length):
            # Get causal mask
            causal_mask = self._generate_causal_mask(generated.shape[1], device)
            
            # Embed current tokens
            gloss_embeds = self.gloss_embedding(generated)
            
            # Decode
            decoder_output = self.decoder(
                gloss_embeds,
                text_hidden,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_mask,
            )
            
            # Get next token
            next_token_logits = self.output_projection(decoder_output[:, -1:, :])
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have EOS
            if (generated == self.eos_token_id).any(dim=1).all():
                break
        
        return {
            "generated_ids": generated,
            "logits": self.output_projection(
                self.gloss_embedding(generated) @ text_hidden.transpose(-2, -1)
            ),
        }
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Generate causal attention mask for decoder."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def decode_gloss(self, gloss_ids: Tensor, idx_to_gloss: Dict[int, str]) -> List[str]:
        """Convert gloss IDs to gloss strings."""
        gloss_sequences = []
        for seq in gloss_ids:
            glosses = []
            for idx in seq:
                if idx.item() == self.eos_token_id:
                    break
                if idx.item() != self.sos_token_id and idx.item() != self.pad_token_id:
                    glosses.append(idx_to_gloss.get(idx.item(), "<UNK>"))
            gloss_sequences.append(" ".join(glosses))
        return gloss_sequences


# =============================================================================
# STAGE 3: GLOSS-TO-POSE GENERATOR
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class GlossToPoseGenerator(nn.Module):
    """
    Stage 3: Generate skeleton poses from gloss sequence.
    Uses a transformer-based motion generator.
    """
    
    def __init__(
        self,
        gloss_vocab_size: int = 1000,
        gloss_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        pose_dim: int = 543,  # 33 body + 21*2 hands + 468 face
        pose_coords: int = 3,  # x, y, z
        max_frames: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.gloss_vocab_size = gloss_vocab_size
        self.gloss_embed_dim = gloss_embed_dim
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim
        self.pose_coords = pose_coords
        self.max_frames = max_frames
        
        # Gloss embedding
        self.gloss_embedding = nn.Embedding(gloss_vocab_size, gloss_embed_dim)
        
        # Project gloss embedding to hidden dim
        self.gloss_projection = nn.Linear(gloss_embed_dim, hidden_dim)
        
        # Duration predictor (predict number of frames for each gloss)
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),  # Ensure positive duration
        )
        
        # Positional encoding for output sequence
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_frames)
        
        # Transformer decoder for pose generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Learnable query embeddings for pose sequence
        self.query_embedding = nn.Parameter(torch.randn(1, max_frames, hidden_dim))
        
        # Output projection to pose coordinates
        self.pose_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, pose_dim * pose_coords),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(
        self,
        gloss_ids: Tensor,
        gloss_mask: Optional[Tensor] = None,
        target_frames: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Generate skeleton poses from gloss sequence.
        
        Args:
            gloss_ids: [batch_size, gloss_seq_len] gloss token indices
            gloss_mask: [batch_size, gloss_seq_len] mask for valid glosses
            target_frames: number of frames to generate (default: self.max_frames)
        
        Returns:
            Dictionary with generated poses and auxiliary outputs
        """
        batch_size = gloss_ids.shape[0]
        device = gloss_ids.device
        
        # Embed glosses
        gloss_embeds = self.gloss_embedding(gloss_ids)
        gloss_embeds = self.gloss_projection(gloss_embeds)  # [batch, gloss_len, hidden]
        
        # Predict duration for each gloss
        durations = self.duration_predictor(gloss_embeds).squeeze(-1)  # [batch, gloss_len]
        
        # Use target_frames or default
        num_frames = target_frames if target_frames is not None else self.max_frames
        
        # Create query embeddings
        queries = self.query_embedding[:, :num_frames, :].expand(batch_size, -1, -1)
        queries = self.pos_encoding(queries)  # [batch, num_frames, hidden]
        
        # Create masks
        tgt_mask = self._generate_causal_mask(num_frames, device)
        memory_mask = None
        if gloss_mask is not None:
            memory_mask = (gloss_mask == 0)
        
        # Decode to generate pose sequence
        decoder_output = self.decoder(
            queries,
            gloss_embeds,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
        )  # [batch, num_frames, hidden]
        
        # Project to pose coordinates
        pose_flat = self.pose_projection(decoder_output)  # [batch, num_frames, pose_dim * pose_coords]
        poses = pose_flat.view(batch_size, num_frames, self.pose_dim, self.pose_coords)
        
        return {
            "poses": poses,
            "durations": durations,
            "hidden_states": decoder_output,
        }
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# =============================================================================
# STAGE 4: POSE REFINEMENT (ST-GCN)
# =============================================================================

class GraphConvolution(nn.Module):
    """Graph convolution layer for skeleton processing."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: [batch, nodes, features]
            adj: [nodes, nodes] adjacency matrix
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class STGCNBlock(nn.Module):
    """Spatio-Temporal Graph Convolutional Network block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (9, 1),
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        # Spatial graph convolution
        self.gcn = GraphConvolution(in_channels, out_channels)
        
        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.Conv1d(
                out_channels, out_channels,
                kernel_size=kernel_size[0],
                stride=stride,
                padding=kernel_size[0] // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: [batch, time, nodes, channels]
            adj: [nodes, nodes] adjacency matrix
        """
        batch_size, time, nodes, channels = x.shape
        
        # Apply GCN for each time step
        x_gcn = []
        for t in range(time):
            x_t = x[:, t, :, :]  # [batch, nodes, channels]
            x_t = self.gcn(x_t, adj)  # [batch, nodes, out_channels]
            x_gcn.append(x_t)
        x_gcn = torch.stack(x_gcn, dim=1)  # [batch, time, nodes, out_channels]
        
        # Reshape for TCN: [batch, channels, time, nodes] -> [batch, channels*time, nodes]
        x_gcn = x_gcn.permute(0, 3, 1, 2)  # [batch, out_channels, time, nodes]
        x_gcn = x_gcn.reshape(batch_size, -1, time * nodes)
        
        # Apply TCN
        x_tcn = self.tcn(x_gcn[:, :x_gcn.shape[1]//time, :time].reshape(batch_size, -1, time))
        
        # Reshape back
        x_out = x_tcn.reshape(batch_size, -1, time)
        x_out = x_out.permute(0, 2, 1)  # [batch, time, channels]
        
        # Residual connection
        x_res = self.residual(x.permute(0, 3, 1, 2).reshape(batch_size, -1, time * nodes)[:, :channels, :time].reshape(batch_size, -1, time))
        
        return self.dropout(x_out + x_res)


class PoseRefiner(nn.Module):
    """
    Stage 4: Refine generated poses using ST-GCN.
    Applies smoothing and kinematic constraints.
    """
    
    def __init__(
        self,
        pose_dim: int = 543,
        pose_coords: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 9,
        temporal_kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pose_dim = pose_dim
        self.pose_coords = pose_coords
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(pose_coords, hidden_dim)
        
        # Create adjacency matrix for skeleton
        self.register_buffer('adj', self._create_adjacency_matrix())
        
        # ST-GCN blocks
        self.blocks = nn.ModuleList([
            STGCNBlock(
                hidden_dim, hidden_dim,
                kernel_size=(temporal_kernel, 1),
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pose_coords),
            nn.Sigmoid(),
        )
        
        # Smoothing layer
        self.smooth_conv = nn.Conv1d(
            pose_dim * pose_coords,
            pose_dim * pose_coords,
            kernel_size=5,
            padding=2,
            groups=pose_dim,  # Separate smoothing per keypoint
        )
    
    def _create_adjacency_matrix(self) -> Tensor:
        """Create adjacency matrix for skeleton graph."""
        # Simplified adjacency for demonstration
        # In practice, use MediaPipe skeleton connections
        adj = torch.eye(self.pose_dim)
        
        # Add body connections (simplified)
        body_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27),
            (24, 26), (26, 28),
        ]
        
        for i, j in body_connections:
            if i < self.pose_dim and j < self.pose_dim:
                adj[i, j] = 1
                adj[j, i] = 1
        
        # Normalize adjacency matrix
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj = adj / degree
        
        return adj
    
    def forward(self, poses: Tensor) -> Dict[str, Tensor]:
        """
        Refine poses using ST-GCN.
        
        Args:
            poses: [batch, time, nodes, coords] skeleton poses
        
        Returns:
            Dictionary with refined poses
        """
        batch_size, time, nodes, coords = poses.shape
        
        # Project to hidden dimension
        x = self.input_proj(poses)  # [batch, time, nodes, hidden]
        
        # Apply ST-GCN blocks
        for block in self.blocks:
            x = block(x, self.adj)
        
        # Project back to coordinates
        refined = self.output_proj(x)  # [batch, time, nodes, coords]
        
        # Apply temporal smoothing
        refined_flat = refined.permute(0, 2, 3, 1).reshape(batch_size, nodes * coords, time)
        smoothed = self.smooth_conv(refined_flat)
        smoothed = smoothed.reshape(batch_size, nodes, coords, time).permute(0, 3, 1, 2)
        
        return {
            "refined_poses": smoothed,
            "residual": smoothed - poses,
        }


# =============================================================================
# COMPLETE TEXT-TO-SIGN MODEL
# =============================================================================

class TextToSignModel(nn.Module):
    """
    Complete Text-to-Sign Translation Model.
    5-Stage Pipeline: Text → Gloss → Pose → Refinement → Output
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Stage 1 & 2: Text encoder and Text-to-Gloss
        self.text_encoder = TextEncoder(
            vocab_size=config.get("vocab_size", 10000),
            embedding_dim=config.get("text_embedding_dim", 768),
            hidden_dim=config.get("text_hidden_dim", 768),
            num_layers=config.get("text_encoder_layers", 4),
            num_heads=config.get("num_heads", 8),
            max_seq_length=config.get("max_text_length", 128),
            dropout=config.get("dropout", 0.1),
        )
        
        self.text_to_gloss = TextToGlossTranslator(
            text_encoder=self.text_encoder,
            gloss_vocab_size=config.get("gloss_vocab_size", 1000),
            hidden_dim=config.get("gloss_hidden_dim", 512),
            num_encoder_layers=config.get("gloss_encoder_layers", 4),
            num_decoder_layers=config.get("gloss_decoder_layers", 4),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
        )
        
        # Stage 3: Gloss-to-Pose
        self.gloss_to_pose = GlossToPoseGenerator(
            gloss_vocab_size=config.get("gloss_vocab_size", 1000),
            gloss_embed_dim=config.get("gloss_embed_dim", 256),
            hidden_dim=config.get("pose_hidden_dim", 512),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("pose_layers", 6),
            pose_dim=config.get("pose_dim", 543),
            pose_coords=config.get("pose_coords", 3),
            max_frames=config.get("max_frames", 60),
            dropout=config.get("dropout", 0.1),
        )
        
        # Stage 4: Pose Refinement
        self.pose_refiner = PoseRefiner(
            pose_dim=config.get("pose_dim", 543),
            pose_coords=config.get("pose_coords", 3),
            hidden_dim=config.get("refine_hidden_dim", 256),
            num_layers=config.get("refine_layers", 9),
            temporal_kernel=config.get("temporal_kernel", 9),
            dropout=config.get("dropout", 0.1),
        )
        
        # Loss functions
        self.gloss_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.pose_loss = nn.MSELoss()
        self.smoothness_loss = nn.L1Loss()
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        gloss_ids: Optional[Tensor] = None,
        target_poses: Optional[Tensor] = None,
        teacher_forcing: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Complete forward pass through all stages.
        
        Args:
            input_ids: [batch, text_len] text token indices
            attention_mask: [batch, text_len] text attention mask
            gloss_ids: [batch, gloss_len] gloss token indices (for training)
            target_poses: [batch, time, nodes, coords] target poses (for training)
            teacher_forcing: use teacher forcing for gloss generation
        
        Returns:
            Dictionary with outputs from each stage and losses
        """
        outputs = {}
        
        # Stage 1-2: Text → Gloss
        if teacher_forcing and gloss_ids is not None:
            gloss_outputs = self.text_to_gloss(input_ids, attention_mask, gloss_ids)
        else:
            gloss_outputs = self.text_to_gloss(input_ids, attention_mask, max_length=20)
        
        outputs["gloss_outputs"] = gloss_outputs
        
        # Get gloss IDs for pose generation
        if gloss_ids is not None:
            gen_gloss_ids = gloss_ids
        else:
            gen_gloss_ids = gloss_outputs.get("generated_ids", gloss_ids)
        
        # Stage 3: Gloss → Pose
        if gen_gloss_ids is not None:
            pose_outputs = self.gloss_to_pose(gen_gloss_ids)
            outputs["pose_outputs"] = pose_outputs
            
            # Stage 4: Refine poses
            refined_outputs = self.pose_refiner(pose_outputs["poses"])
            outputs["refined_outputs"] = refined_outputs
            outputs["final_poses"] = refined_outputs["refined_poses"]
        
        # Calculate losses if targets provided
        if target_poses is not None and "final_poses" in outputs:
            outputs["losses"] = self._compute_losses(
                outputs, gloss_ids, target_poses
            )
        
        return outputs
    
    def _compute_losses(
        self,
        outputs: Dict[str, Tensor],
        gloss_ids: Optional[Tensor],
        target_poses: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute all losses for training."""
        losses = {}
        
        # Gloss loss
        if gloss_ids is not None and "logits" in outputs["gloss_outputs"]:
            logits = outputs["gloss_outputs"]["logits"]
            targets = gloss_ids[:, 1:]  # Shift targets
            if logits.shape[1] < targets.shape[1]:
                targets = targets[:, :logits.shape[1]]
            losses["gloss_loss"] = self.gloss_loss(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )
        
        # Pose reconstruction loss
        if "final_poses" in outputs:
            pred_poses = outputs["final_poses"]
            target_poses_truncated = target_poses[:, :pred_poses.shape[1]]
            losses["pose_loss"] = self.pose_loss(pred_poses, target_poses_truncated)
            
            # Smoothness loss (penalize large frame-to-frame changes)
            if pred_poses.shape[1] > 1:
                motion = pred_poses[:, 1:] - pred_poses[:, :-1]
                losses["smoothness_loss"] = self.smoothness_loss(
                    motion, torch.zeros_like(motion)
                )
        
        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss
        
        return losses
    
    def generate(
        self,
        text: str,
        tokenizer,
        idx_to_gloss: Dict[int, str],
        max_frames: int = 60,
    ) -> Dict[str, Any]:
        """
        Generate sign language from text input.
        
        Args:
            text: input text string
            tokenizer: tokenizer for text processing
            idx_to_gloss: mapping from gloss index to gloss string
            max_frames: maximum number of frames to generate
        
        Returns:
            Dictionary with gloss sequence and generated poses
        """
        self.eval()
        
        with torch.no_grad():
            # Tokenize text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_text_length", 128),
            )
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            # Stage 1-2: Text → Gloss
            gloss_outputs = self.text_to_gloss(
                input_ids, attention_mask, max_length=20
            )
            
            generated_gloss_ids = gloss_outputs.get("generated_ids", None)
            gloss_text = None
            if generated_gloss_ids is not None:
                gloss_text = self.text_to_gloss.decode_gloss(
                    generated_gloss_ids, idx_to_gloss
                )
            
            # Stage 3-4: Gloss → Pose → Refine
            if generated_gloss_ids is not None:
                pose_outputs = self.gloss_to_pose(
                    generated_gloss_ids, target_frames=max_frames
                )
                refined_outputs = self.pose_refiner(pose_outputs["poses"])
                final_poses = refined_outputs["refined_poses"]
            else:
                # Generate default poses
                final_poses = torch.zeros(1, max_frames, 543, 3)
            
            return {
                "text": text,
                "gloss_ids": generated_gloss_ids,
                "gloss_text": gloss_text,
                "poses": final_poses,
            }
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, config: Dict[str, Any]):
        """Load model from checkpoint."""
        model = cls(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_memory_size(model: nn.Module) -> float:
    """Get model memory size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


if __name__ == "__main__":
    # Test the model
    print("Testing TextToSignModel...")
    
    config = {
        "vocab_size": 10000,
        "text_embedding_dim": 768,
        "text_hidden_dim": 768,
        "text_encoder_layers": 4,
        "gloss_vocab_size": 100,
        "gloss_hidden_dim": 512,
        "gloss_encoder_layers": 4,
        "gloss_decoder_layers": 4,
        "gloss_embed_dim": 256,
        "pose_hidden_dim": 512,
        "pose_layers": 6,
        "refine_hidden_dim": 256,
        "refine_layers": 9,
        "pose_dim": 543,
        "pose_coords": 3,
        "num_heads": 8,
        "max_frames": 60,
        "max_text_length": 128,
        "temporal_kernel": 9,
        "dropout": 0.1,
    }
    
    model = TextToSignModel(config)
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Model memory: {get_model_memory_size(model):.2f} MB")
    
    # Test forward pass
    batch_size = 2
    text_len = 20
    gloss_len = 10
    frames = 60
    nodes = 543
    coords = 3
    
    input_ids = torch.randint(0, 1000, (batch_size, text_len))
    attention_mask = torch.ones(batch_size, text_len)
    gloss_ids = torch.randint(0, 100, (batch_size, gloss_len))
    target_poses = torch.rand(batch_size, frames, nodes, coords)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        gloss_ids=gloss_ids,
        target_poses=target_poses,
        teacher_forcing=True,
    )
    
    print(f"\nOutput keys: {outputs.keys()}")
    if "losses" in outputs:
        print(f"Total loss: {outputs['losses']['total_loss'].item():.4f}")
    if "final_poses" in outputs:
        print(f"Final poses shape: {outputs['final_poses'].shape}")
    
    print("\n✅ Model test passed!")
