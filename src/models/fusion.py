"""Cross-modal fusion module."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.v2a_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.a2v_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, vis_tokens: torch.Tensor, aud_tokens: torch.Tensor) -> torch.Tensor:
        vis_attended, _ = self.v2a_attn(query=vis_tokens, key=aud_tokens, value=aud_tokens)
        vis_out = self.norm_v(vis_tokens + vis_attended)

        aud_attended, _ = self.a2v_attn(query=aud_tokens, key=vis_tokens, value=vis_tokens)
        aud_out = self.norm_a(aud_tokens + aud_attended)

        vis_global = vis_out[:, 0, :]
        aud_global = aud_out[:, 0, :]
        fused = self.ffn(torch.cat([vis_global, aud_global], dim=-1))
        return self.norm_out(fused)
