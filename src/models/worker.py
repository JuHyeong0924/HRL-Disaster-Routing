
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm

class WorkerLSTM(nn.Module):
    def __init__(self, node_dim: int = 8, hidden_dim: int = 256, num_layers: int = 3, 
                 dropout: float = 0.2, edge_dim: int = 5, num_scorer_heads: int = 4,
                 ablation_config: dict = None):
        """
        Worker: Local Navigation with Memory.
        [v4.1] 아키텍처 및 State 개편:
          ① Node State (7-Dim): is_curr, is_subgoal, is_final, hop_to_subgoal, hop_to_final, global_heading_x, global_heading_y
             (불필요하게 복제되던 time_to_go 제거)
          ② Global State: time_to_go는 GATv2 통과 후 LSTM 입력에 직접 병합하여 효율성 극대화
          ③ GraphNorm 및 Single-head Scorer 적용 (최적화)
        
        Args:
            node_dim: 노드 피처 차원 (7)
            hidden_dim: 은닉층 차원
            num_layers: GATv2 레이어 수
            dropout: Dropout 비율
            edge_dim: 엣지 피처 차원 (3: [length, capacity, speed] 정규화)
            num_scorer_heads: 사용 안 함 (1로 강제)
            ablation_config: Ablation 실험 설정 딕셔너리
        """
        super(WorkerLSTM, self).__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_dim = node_dim
        
        # Ablation 설정: 기본값은 모두 활성화
        self.ablation_config = ablation_config or {}
        self.use_residual = self.ablation_config.get("use_residual", True)
        self.use_graph_norm = self.ablation_config.get("use_graph_norm", True)
        self.use_lstm = self.ablation_config.get("use_lstm", True)
        
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
        
        # 레이어별 GraphNorm (Residual 후 안정화, Over-smoothing 방지)
        if self.use_graph_norm:
            self.graph_norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        else:
            # [A3] GraphNorm 비활성화 시 Identity 대체 (인터페이스 유지)
            self.graph_norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
            
        # 2. Temporal Memory
        # time_to_go 제거 여부에 따라 LSTM 입력 차원 결정
        _time_dim = 0 if self.ablation_config.get("remove_time_to_go", False) else 1
        
        if self.use_lstm:
            # 기존: LSTM으로 시간적 기억 유지
            self.lstm = nn.LSTMCell(hidden_dim + _time_dim, hidden_dim)
        else:
            # [A5] LSTM 제거: 단순 Linear 투영으로 대체 (시간적 기억 없음)
            self.temporal_proj = nn.Sequential(
                nn.Linear(hidden_dim + _time_dim, hidden_dim),
                nn.ReLU(),
            )
        
        # 3. Decision Head: Single-head Scorer (다이어트)
        self.num_scorer_heads = 1
        self.scorer_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # [Node_Emb, LSTM_H]
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
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
                         batch: torch.Tensor, time_to_go: torch.Tensor = None,
                         neighbors_mask=None, 
                         detach_spatial: bool = False, edge_attr: torch.Tensor = None):
        """
        단일 스텝 추론 (Autoregressive or RL Rollout)
        
        Args:
            x: [N, node_dim] 노드 피처
            edge_index: [2, E] 엣지 인덱스
            h_state: [Batch, hidden_dim]
            c_state: [Batch, hidden_dim]
            batch: [N] 각 노드의 그래프 소속 인덱스
            time_to_go: [Batch, 1] 남은 시간 비율 (Global Feature), S3에서 None
            neighbors_mask: (Optional) 방문 불가능한 노드 마스킹
            detach_spatial: 역전파를 GNN에서 끊을지 여부
            edge_attr: [E, edge_dim] (None이면 edge feature 미사용)
        """
        if x.size(1) < self.node_dim:
            pad = torch.zeros(x.size(0), self.node_dim - x.size(1), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        # === Spatial Encoding (GATv2 + 조건부 Residual/GraphNorm) ===
        if detach_spatial:
            # RL 학습: GATv2는 no_grad로 실행 → VRAM 절약
            with torch.no_grad():
                h = self._forward_gnn(x, edge_index, batch, edge_attr, training=False)
            h = h.detach()  # gradient 그래프에서 완전 분리
        else:
            # SL 학습: End-to-End 직접 역전파
            h = self._forward_gnn(x, edge_index, batch, edge_attr, training=self.training)
        
        # === Temporal Memory ===
        is_current = x[:, 0].bool()
        
        # [Safety] Prevent crash if no current node is found
        if is_current.sum() == 0:
            return torch.zeros(x.size(0), device=x.device), h_state, c_state, torch.zeros(1, device=x.device)

        curr_emb_raw = h[is_current]
        
        # [Safety] Ensure batch alignment
        batch_size = h_state.size(0)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        if curr_emb_raw.size(0) != batch_size:
             # [Perf] 벡터화: for-loop 대신 텐서 인덱싱으로 GPU-CPU 동기화 병목 제거
             curr_emb = torch.zeros((batch_size, self.hidden_dim), device=x.device, dtype=curr_emb_raw.dtype)
             found_batch_idx = batch[is_current]
             valid_mask = found_batch_idx < batch_size
             curr_emb[found_batch_idx[valid_mask]] = curr_emb_raw[valid_mask]
        else:
             curr_emb = curr_emb_raw

        # time_to_go 병합 (S3 ablation: 제거 시 병합하지 않음)
        if time_to_go is not None and not self.ablation_config.get("remove_time_to_go", False):
            curr_emb_input = torch.cat([curr_emb, time_to_go], dim=1)  # [Batch, hidden_dim + 1]
        else:
            curr_emb_input = curr_emb  # [Batch, hidden_dim]

        if self.use_lstm:
            # 기존: LSTM으로 시간적 기억 유지
            h_next, c_next = self.lstm(curr_emb_input, (h_state, c_state))
        else:
            # [A5] LSTM 제거: Linear 투영 (상태 비유지, 매 스텝 독립 판단)
            h_next = self.temporal_proj(curr_emb_input)
            c_next = c_state  # dummy, 변경 없음
        
        # Value Prediction (Critic)
        value = self.critic(h_next)
        
        # ② Multi-head Scorer: 각 head의 점수를 평균하여 앙상블
        context_expanded = h_next[batch]
        combined = torch.cat([h, context_expanded], dim=1)
        
        # 각 head에서 독립적으로 스코어링 후 평균
        head_scores = [head(combined) for head in self.scorer_heads]
        scores = torch.stack(head_scores, dim=0).mean(dim=0).squeeze(-1)
        
        return scores, h_next, c_next, value
    
    def _forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor,
                     batch: torch.Tensor, edge_attr: torch.Tensor = None,
                     training: bool = True) -> torch.Tensor:
        """GNN Forward (Architecture Ablation 조건부 적용)."""
        h = x
        
        # 첫 번째 레이어: input projection + GATv2
        if self.use_residual:
            h_residual = self.input_proj(h)
        
        h = self.convs[0](h, edge_index, edge_attr=edge_attr)
        
        if training:
            h = F.dropout(F.elu(h), p=self.dropout, training=True)
        else:
            h = F.elu(h)
        
        # [A2] Residual 적용 여부
        if self.use_residual:
            h = h + h_residual
        
        # [A3] GraphNorm 적용 여부 (Identity면 no-op)
        if self.use_graph_norm:
            h = self.graph_norms[0](h, batch)
        
        # 나머지 레이어
        for i in range(1, self.num_layers):
            if self.use_residual:
                h_residual = h
            
            h = self.convs[i](h, edge_index, edge_attr=edge_attr)
            
            if training:
                h = F.dropout(F.elu(h), p=self.dropout, training=True)
            else:
                h = F.elu(h)
            
            if self.use_residual:
                h = h + h_residual
            
            if self.use_graph_norm:
                h = self.graph_norms[i](h, batch)
        
        return h
