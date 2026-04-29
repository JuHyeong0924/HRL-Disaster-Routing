import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch_geometric.nn import GATv2Conv, GraphNorm

class Worker(nn.Module):
    """
    Phase 1: HRL Worker (Ablation 최적 구조 반영)
    - 4-Dim State: [is_curr, is_tgt, is_next_zone, hop_dist]
    - Spatial: 2-Layer GATv2 + GraphNorm + Residual
    - Temporal: LSTM 제거 (Linear 투영)
    - Checkpointing: torch.utils.checkpoint 적용 → VRAM 절약
    """
    def __init__(self, node_dim: int = 4, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2, use_checkpoint: bool = False):
        super(Worker, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint  # VRAM 절약용 Gradient Checkpointing
        
        # 1. Spatial Encoder
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(node_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
            
        self.graph_norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        
        # 2. Temporal (LSTM 대신 단순 투영)
        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 3. Policy Head (Scorer)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 4. Value Head (Critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def _forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """GATv2 2-Layer 공간 인코더 (Residual + GraphNorm).
        
        use_checkpoint=True 시 각 레이어를 torch.utils.checkpoint로 감싸
        순전파 중간 활성값을 저장하지 않아 VRAM을 ~3배 절약.
        역전파 시 해당 레이어를 재계산하는 방식으로 동작.
        """
        def _layer0(x_in: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
            # 첫 번째 GATv2 레이어: 4 → 256
            h = self.convs[0](x_in, ei)
            h = self.graph_norms[0](h, batch)
            return torch.relu(h + self.input_proj(x_in))  # Residual

        def _layer_n(h_in: torch.Tensor, ei: torch.Tensor, i: int) -> torch.Tensor:
            # i번째 GATv2 레이어: 256 → 256
            h_next = self.convs[i](h_in, ei)
            h_next = self.graph_norms[i](h_next, batch)
            return torch.relu(h_next + h_in)  # Residual

        if self.use_checkpoint:
            # 중간 활성값 저장 안 함 → VRAM 절약 (속도 약 20% 희생)
            h = grad_ckpt(_layer0, x, edge_index, use_reentrant=False)
            for i in range(1, self.num_layers):
                h = grad_ckpt(_layer_n, h, edge_index, i, use_reentrant=False)
        else:
            h = _layer0(x, edge_index)
            for i in range(1, self.num_layers):
                h = _layer_n(h, edge_index, i)

        return h

    def forward(self, x, edge_index, batch, neighbors_mask=None, detach_spatial=False):
        """
        Args:
            x: [N, 4] node features
            edge_index: [2, E] edge indices
            batch: [N] graph assignment
            neighbors_mask: [N] action mask
        Returns:
            action_probs: [N] softmax probabilities over masked nodes
            value: [1] state value
        """
        if detach_spatial:
            with torch.no_grad():
                h = self._forward_gnn(x, edge_index, batch)
            h = h.detach()
        else:
            h = self._forward_gnn(x, edge_index, batch)
            
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
        
        # Policy Scoring
        # Broadcast h_t to all nodes in the respective graphs
        if batch is not None:
            h_t_expanded = h_t[batch]
        else:
            h_t_expanded = h_t.expand(x.size(0), -1)
            
        scorer_input = torch.cat([h, h_t_expanded], dim=-1)
        logits = self.scorer(scorer_input).squeeze(-1) # [N]
        
        # Apply mask
        if neighbors_mask is not None:
            logits = logits.masked_fill(neighbors_mask == 0, float('-inf'))
            
        # Value estimate
        value = self.critic(h_t) # [Batch, 1]
        
        # Softmax over actions (per graph in batch)
        if batch is not None:
            from torch_geometric.utils import softmax as pyg_softmax
            probs = pyg_softmax(logits, batch)
        else:
            probs = torch.softmax(logits, dim=0)
            
        return probs, value, h_t
