"""
Manager v2: 비자기회귀 단일 서브골 예측 모델 (ReactiveManager)

Worker와 통일된 GATv2 + Dual Head(Actor/Critic) 아키텍처.
Transformer Decoder를 제거하고, 매 턴마다 K-hop 반경 내에서
서브골 1개를 선택하는 비자기회귀(Non-autoregressive) 방식.

핵심 설계:
- GATv2Conv × 2L + GraphNorm + Residual (Worker와 동일)
- Actor Head: h_curr ∥ h_goal ∥ h_candidate → MLP → Score
- Critic Head: h_curr ∥ h_goal → MLP → V(s)
- K-hop 마스킹으로 행동 공간 제한
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm
from typing import Tuple, Optional


class ReactiveManager(nn.Module):
    """비자기회귀 단일 서브골 예측 Manager.

    매 턴마다 전체 그래프를 GATv2로 인코딩한 후,
    K-hop 반경 내의 후보 노드 중 최적의 서브골 1개를 선택한다.
    PPO 학습을 위해 Actor/Critic Dual Head 구조를 갖는다.

    Args:
        node_dim: 노드 피처 차원 (S7 기준 4: is_curr, is_tgt, hop_dist, degree)
        hidden_dim: GNN 및 MLP 히든 차원
        num_layers: GATv2 레이어 수
        gat_heads: GATv2 어텐션 헤드 수
        dropout: 드롭아웃 비율
    """

    def __init__(
        self,
        node_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── GATv2 Spatial Encoder (Worker와 동일 구조) ──
        self.convs = nn.ModuleList()
        # 첫 레이어: node_dim → hidden_dim
        self.convs.append(
            GATv2Conv(node_dim, hidden_dim, heads=gat_heads, concat=False, dropout=dropout)
        )
        # 입력 차원 맞춤용 프로젝션 (Residual Connection용)
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # 나머지 레이어: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, concat=False, dropout=dropout)
            )

        # GraphNorm: 레이어별 정규화 (Ablation v1에서 필수 컴포넌트로 확인됨)
        self.graph_norms = nn.ModuleList(
            [GraphNorm(hidden_dim) for _ in range(num_layers)]
        )

        # ── Actor Head: 후보 서브골 점수 산출 ──
        # 입력: [h_curr ∥ h_goal ∥ h_candidate] = 3 * hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # ── Critic Head: 상태 가치 V(s) 추정 ──
        # 입력: [h_curr ∥ h_goal] = 2 * hidden_dim
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """GATv2 공간 인코더 (Residual + GraphNorm).

        Args:
            x: [N_total, node_dim] 노드 피처 (배치 내 모든 그래프의 노드)
            edge_index: [2, E_total] 엣지 인덱스
            batch: [N_total] 각 노드가 속한 그래프 인덱스

        Returns:
            h: [N_total, hidden_dim] 인코딩된 노드 임베딩
        """
        # Layer 0: node_dim → hidden_dim (input_proj로 Residual)
        h = self.convs[0](x, edge_index)
        if batch is not None:
            h = self.graph_norms[0](h, batch)
        else:
            h = self.graph_norms[0](h)
        h = torch.relu(h + self.input_proj(x))

        # Layer 1+: hidden_dim → hidden_dim (직접 Residual)
        for i in range(1, self.num_layers):
            h_prev = h
            h = self.convs[i](h, edge_index)
            if batch is not None:
                h = self.graph_norms[i](h, batch)
            else:
                h = self.graph_norms[i](h)
            h = torch.relu(h + h_prev)

        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        current_idx: int,
        goal_idx: int,
        candidate_mask: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """단일 그래프에 대한 서브골 선택 확률 및 상태 가치 출력.

        Args:
            x: [N, node_dim] 노드 피처
            edge_index: [2, E] 엣지 인덱스
            current_idx: 현재 위치 노드 인덱스 (스칼라)
            goal_idx: 최종 목적지 노드 인덱스 (스칼라)
            candidate_mask: [N] K-hop 내 후보 노드 마스크 (1=후보, 0=비후보)
            batch: [N] 그래프 배치 인덱스 (단일 그래프면 None)

        Returns:
            probs: [N] 각 노드의 서브골 선택 확률 (마스킹 적용 후 Softmax)
            value: [1] 현재 상태의 가치 추정값
            logits: [N] 마스킹 전 raw logits (PPO ratio 계산용)
        """
        # 1. GNN 인코딩
        h = self._encode(x, edge_index, batch)  # [N, hidden_dim]

        # 2. 현재 위치 및 목적지 임베딩 추출
        h_curr = h[current_idx].unsqueeze(0)  # [1, hidden_dim]
        h_goal = h[goal_idx].unsqueeze(0)     # [1, hidden_dim]

        # 3. Actor: 모든 노드에 대한 점수 계산
        N = h.size(0)
        # h_curr, h_goal을 N개 노드로 브로드캐스트
        h_curr_exp = h_curr.expand(N, -1)  # [N, hidden_dim]
        h_goal_exp = h_goal.expand(N, -1)  # [N, hidden_dim]
        # 각 노드(후보 서브골)별 점수 입력: [h_curr ∥ h_goal ∥ h_node]
        actor_input = torch.cat([h_curr_exp, h_goal_exp, h], dim=-1)  # [N, 3*hidden_dim]
        logits = self.actor(actor_input).squeeze(-1)  # [N]

        # 4. K-hop 마스킹 적용
        logits = logits.masked_fill(candidate_mask == 0, float('-inf'))

        # 5. Softmax → 확률 분포
        probs = F.softmax(logits, dim=-1)

        # 6. Critic: 상태 가치 추정
        critic_input = torch.cat([h_curr, h_goal], dim=-1)  # [1, 2*hidden_dim]
        value = self.critic(critic_input).squeeze(-1)        # [1]

        return probs, value, logits

    def select_action(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        current_idx: int,
        goal_idx: int,
        candidate_mask: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """서브골 1개를 선택하고 PPO 학습에 필요한 정보를 반환.

        Args:
            deterministic: True면 가장 높은 확률의 노드 선택 (평가용)

        Returns:
            action: 선택된 서브골 노드 인덱스 (int)
            log_prob: 선택된 액션의 로그 확률 [1]
            value: 상태 가치 추정값 [1]
            entropy: 정책 엔트로피 [1]
        """
        probs, value, logits = self.forward(
            x, edge_index, current_idx, goal_idx, candidate_mask, batch
        )

        # 유효한 후보가 없는 경우 방어
        if (candidate_mask == 0).all():
            # 현재 위치를 반환 (에피소드 종료 트리거용)
            return current_idx, torch.tensor(0.0, device=x.device), value, torch.tensor(0.0, device=x.device)

        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return int(action.item()), log_prob, value, entropy
