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

    def forward(self, vis_embed: torch.Tensor, aud_embed: torch.Tensor) -> torch.Tensor:
        vis_seq = vis_embed.unsqueeze(1)
        aud_seq = aud_embed.unsqueeze(1)

        vis_attended, _ = self.v2a_attn(query=vis_seq, key=aud_seq, value=aud_seq)
        vis_out = self.norm_v(vis_seq + vis_attended).squeeze(1)

        aud_attended, _ = self.a2v_attn(query=aud_seq, key=vis_seq, value=vis_seq)
        aud_out = self.norm_a(aud_seq + aud_attended).squeeze(1)

        fused = self.ffn(torch.cat([vis_out, aud_out], dim=-1))
        return self.norm_out(fused)

