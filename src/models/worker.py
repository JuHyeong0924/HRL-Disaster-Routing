import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool

class Worker(nn.Module):
    """
    HRL Worker (Model D 지원)
    - Base (5-dim): [is_curr, is_tgt, zone_info, dist_to_tgt, dist_to_next_z]
    - +is_visited (6-dim): 방문 노드 이력 → 순환 방지
    - +node_damage (7-dim): 노드 단위 재해 여부 확인 (Preemptive Detour)
    - +global_pool: Critic에 전역 그래프 맥락 주입
    - Spatial: 2-Layer GATv2 + GraphNorm + Residual
    - Temporal: Linear 투영
    """
    def __init__(self, node_dim: int = 7, hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.2, use_checkpoint: bool = False,
                 use_jk_net: bool = False):
        super(Worker, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.use_jk_net = use_jk_net
        # 1. Spatial Encoder
        self.convs = nn.ModuleList()
        edge_dim = 2  # [length, damage]
        self.convs.append(GATv2Conv(node_dim, hidden_dim, heads=4, concat=False, dropout=dropout, edge_dim=edge_dim))
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout, edge_dim=edge_dim))
            
        self.graph_norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        
        if self.use_jk_net:
            self.jk_proj = nn.Linear(num_layers * hidden_dim, hidden_dim)
        
        # 2. Temporal (LSTM 대신 단순 투영)
        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 3. Policy Head (Actor)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 4. Value Head (Critic) 복원 (PPO 필수)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def _forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """GATv2 공간 인코더 (Residual + GraphNorm + JK-Net)."""
        def _layer0(x_in: torch.Tensor, ei: torch.Tensor, ea: torch.Tensor) -> torch.Tensor:
            h = self.convs[0](x_in, ei, edge_attr=ea)
            h = self.graph_norms[0](h, batch)
            return torch.relu(h + self.input_proj(x_in))  # Residual

        def _layer_n(h_in: torch.Tensor, ei: torch.Tensor, ea: torch.Tensor, i: int) -> torch.Tensor:
            h_next = self.convs[i](h_in, ei, edge_attr=ea)
            h_next = self.graph_norms[i](h_next, batch)
            return torch.relu(h_next + h_in)  # Residual

        h_list = []
        if self.use_checkpoint:
            h = grad_ckpt(_layer0, x, edge_index, edge_attr, use_reentrant=False)
            h_list.append(h)
            for i in range(1, self.num_layers):
                h = grad_ckpt(_layer_n, h, edge_index, edge_attr, i, use_reentrant=False)
                h_list.append(h)
        else:
            h = _layer0(x, edge_index, edge_attr)
            h_list.append(h)
            for i in range(1, self.num_layers):
                h = _layer_n(h, edge_index, edge_attr, i)
                h_list.append(h)

        if self.use_jk_net:
            return self.jk_proj(torch.cat(h_list, dim=-1))
        return h

    def forward(self, x, edge_index, batch, neighbors_mask=None, detach_spatial=False, edge_attr=None):
        """
        Args:
            x: [N, node_dim] node features
            edge_index: [2, E] edge indices
            batch: [N] graph assignment
            neighbors_mask: [N] action mask
        Returns:
            action_probs: [N] softmax probabilities over masked nodes
            logits: [N] raw logits
        """
        # edge_attr is always used
        if detach_spatial:
            with torch.no_grad():
                h = self._forward_gnn(x, edge_index, batch, edge_attr=edge_attr)
            h = h.detach()
        else:
            h = self._forward_gnn(x, edge_index, batch, edge_attr=edge_attr)
            
        # Current node embedding
        is_curr = x[:, 0].bool()
        curr_emb_raw = h[is_curr]
        
        batch_size = 1
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
            
        if curr_emb_raw.size(0) == 0:
            curr_emb = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            curr_emb = curr_emb_raw
            
        # Temporal projection (instead of LSTM)
        h_t = self.temporal_proj(curr_emb)
        
        # Policy Scoring: Broadcast current node's context to all nodes
        if batch is not None:
            temporal_out = h_t[batch]
        else:
            temporal_out = h_t.expand(h.size(0), -1)
            
        combined = torch.cat([h, temporal_out], dim=-1)
        
        # Actor: Masked Softmax Probabilities
        logits = self.scorer(combined).squeeze(-1)
        
        if neighbors_mask is not None:
            logits = logits.masked_fill(neighbors_mask == 0, float('-inf'))
            
        # Softmax over actions (per graph in batch)
        if batch is not None:
            from torch_geometric.utils import softmax as pyg_softmax
            action_probs = pyg_softmax(logits, batch)
        else:
            action_probs = torch.softmax(logits, dim=0)
            
        # Critic: State Value estimation
        if batch is not None:
            pooled = global_mean_pool(combined, batch)
        else:
            pooled = combined.mean(dim=0, keepdim=True)
            
        value = self.value_head(pooled).squeeze(-1) # [B]
            
        return action_probs, value, logits
