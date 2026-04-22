
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class WorkerLSTM(nn.Module):
    def __init__(self, node_dim: int = 9, hidden_dim: int = 256, num_layers: int = 3, 
                 dropout: float = 0.2, edge_dim: int = 5, num_scorer_heads: int = 4):
        """
        Worker: Local Navigation with Memory.
        [v2] 아키텍처 강화:
          ① Residual Connection: GATv2 레이어 간 skip connection (gradient flow 개선)
          ② Multi-head Scorer: 4개 독립 scorer 앙상블 (의사결정 다양화)
          ③ Edge Feature: GATv2 attention에 도로 속성 반영 (edge_dim)
        
        Args:
            node_dim: 노드 피처 차원
                (9: [x, y, is_curr, is_tgt, net_dist, dir_x, dir_y, is_final_target_phase, hop_dist])
            hidden_dim: 은닉층 차원
            num_layers: GATv2 레이어 수
            dropout: Dropout 비율
            edge_dim: 엣지 피처 차원 (SL=1: length, RL Phase2=9: full attrs)
            num_scorer_heads: Multi-head Scorer의 head 수
        """
        super(WorkerLSTM, self).__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_dim = node_dim
        
        # 1. Spatial Encoder (GATv2 + Residual)
        self.convs = nn.ModuleList()
        # 첫 번째 레이어: 차원 변환 (node_dim → hidden_dim)
        self.convs.append(GATv2Conv(node_dim, hidden_dim, heads=4, concat=False, 
                                     dropout=dropout, edge_dim=edge_dim))
        # ① Residual을 위한 projection (첫 레이어 차원 불일치 보정)
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # 나머지 레이어: 동일 차원 (hidden_dim → hidden_dim) → 직접 Residual 가능
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, 
                                         dropout=dropout, edge_dim=edge_dim))
        
        # 레이어별 LayerNorm (Residual 후 안정화)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            
        # 2. Temporal Memory (LSTM)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # 3. Decision Head: ② Multi-head Scorer
        self.num_scorer_heads = num_scorer_heads
        self.scorer_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // num_scorer_heads),  # [Node_Emb, LSTM_H]
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // num_scorer_heads, 1)
            ) for _ in range(num_scorer_heads)
        ])
        
        # 4. Value Head (Critic): [Refactor: Task 2] 2-Layer MLP 고도화
        # Why: 복잡한 PBRS 보상 환경에서 Explained Variance 붕괴 방지
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def predict_next_hop(self, x: torch.Tensor, edge_index: torch.Tensor, 
                         h_state: torch.Tensor, c_state: torch.Tensor, 
                         batch: torch.Tensor, neighbors_mask=None, 
                         detach_spatial: bool = False, edge_attr: torch.Tensor = None):
        """
        Single step prediction with batch support.
        
        Args:
            x: 노드 피처 [B*N, node_dim]
            edge_index: 엣지 인덱스 [2, E]
            h_state: LSTM hidden state [B, H]
            c_state: LSTM cell state [B, H]
            batch: 배치 인덱스 [B*N]
            detach_spatial: True이면 GATv2 출력을 detach (RL 학습 시 VRAM 절약)
            edge_attr: ③ 엣지 피처 [E, edge_dim] (None이면 edge feature 미사용)
        """
        if x.size(1) < self.node_dim:
            pad = torch.zeros(x.size(0), self.node_dim - x.size(1), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        # === Spatial Encoding (GATv2 + Residual) ===
        if detach_spatial:
            # RL 학습: GATv2는 no_grad로 실행 → VRAM 절약
            with torch.no_grad():
                h = x
                # ① 첫 번째 레이어: input projection + GATv2 + Residual
                h_residual = self.input_proj(h)
                h = self.convs[0](h, edge_index, edge_attr=edge_attr)
                h = F.elu(h) + h_residual
                h = self.layer_norms[0](h)
                
                # 나머지 레이어: 직접 Residual
                for i in range(1, self.num_layers):
                    h_residual = h
                    h = self.convs[i](h, edge_index, edge_attr=edge_attr)
                    h = F.elu(h) + h_residual  # ① Skip Connection
                    h = self.layer_norms[i](h)
            h = h.detach()  # gradient 그래프에서 완전 분리
        else:
            # RL / SL 학습: End-to-End 직접 역전파 (속도 우선, VRAM ~13GiB 소모)
            h = x
            
            # 첫 번째 레이어
            h_residual = self.input_proj(h)
            h = self.convs[0](h, edge_index, edge_attr=edge_attr)
            h = F.dropout(F.elu(h), p=self.dropout, training=self.training)
            h = h + h_residual
            h = self.layer_norms[0](h)
            
            # 나머지 레이어
            for i in range(1, self.num_layers):
                h_residual = h
                h = self.convs[i](h, edge_index, edge_attr=edge_attr)
                h = F.dropout(F.elu(h), p=self.dropout, training=self.training)
                h = h + h_residual
                h = self.layer_norms[i](h)
        
        # === Temporal Memory (LSTM) — gradient 유지 ===
        is_current = x[:, 2].bool()
        
        # [Safety] Prevent crash if no current node is found
        if is_current.sum() == 0:
            return torch.zeros(x.size(0), device=x.device), h_state, c_state, torch.zeros(1, device=x.device)

        curr_emb_raw = h[is_current]
        
        # [Safety] Ensure batch alignment
        batch_size = h_state.size(0)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        if curr_emb_raw.size(0) != batch_size:
             # 배치 크기 불일치 시 안전 처리
             curr_emb = torch.zeros((batch_size, self.hidden_dim), device=x.device)
             found_batch_idx = batch[is_current]
             for i, b_idx in enumerate(found_batch_idx):
                 if b_idx < batch_size:
                     curr_emb[b_idx] = curr_emb_raw[i]
        else:
             curr_emb = curr_emb_raw

        h_next, c_next = self.lstm(curr_emb, (h_state, c_state))
        
        # Value Prediction (Critic)
        value = self.critic(h_next)
        
        # ② Multi-head Scorer: 각 head의 점수를 평균하여 앙상블
        context_expanded = h_next[batch]
        combined = torch.cat([h, context_expanded], dim=1)
        
        # 각 head에서 독립적으로 스코어링 후 평균
        head_scores = [head(combined) for head in self.scorer_heads]
        scores = torch.stack(head_scores, dim=0).mean(dim=0).squeeze(-1)
        
        return scores, h_next, c_next, value
