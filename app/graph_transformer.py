from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class GraphTransformerHierarchyModel(nn.Module):
    """Attention-based GNN skeleton for second-stage hierarchy parsing."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, num_heads: int = 4, num_types: int = 10):
        super().__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.type_head = nn.Linear(hidden_dim, num_types)
        self.level_head = nn.Linear(hidden_dim, 5)
        self.parent_query = nn.Linear(hidden_dim, hidden_dim)
        self.parent_key = nn.Linear(hidden_dim, hidden_dim)
        self.overlap_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, node_features: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hidden = self.node_encoder(node_features)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)

        query = self.parent_query(hidden)
        key = self.parent_key(hidden)
        parent_logits = torch.matmul(query, key.transpose(1, 2)) / (hidden.shape[-1] ** 0.5)

        pair_a = hidden.unsqueeze(2).expand(-1, -1, hidden.shape[1], -1)
        pair_b = hidden.unsqueeze(1).expand(-1, hidden.shape[1], -1, -1)
        overlap_logits = self.overlap_head(torch.cat([pair_a, pair_b], dim=-1))

        return {
            "type_logits": self.type_head(hidden),
            "level_logits": self.level_head(hidden),
            "parent_logits": parent_logits,
            "overlap_logits": overlap_logits,
        }


def load_graph_transformer(checkpoint_path: str, input_dim: int) -> GraphTransformerHierarchyModel:
    state = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = state.get("hidden_dim", 128) if isinstance(state, dict) else 128
    layers = state.get("layers", 3) if isinstance(state, dict) else 3
    heads = state.get("heads", 4) if isinstance(state, dict) else 4
    type_names = state.get("type_names", []) if isinstance(state, dict) else []
    model = GraphTransformerHierarchyModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=layers,
        num_heads=heads,
        num_types=max(1, len(type_names)) if type_names else 10,
    )
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    return model
