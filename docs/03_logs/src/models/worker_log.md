# 📝 Worker Network Log (`src/models/worker.py`)

## 1. Intent & Purpose (Detailed Functional Specs)
*   **Role:** 주어진 HRL 그래프 환경에서 현재 노드(`is_curr`)에서 다음 타겟 노드(`is_tgt`)로 이동하기 위한 최적의 이웃 노드(액션) 확률 분포와 상태 가치(Value)를 산출하는 Actor-Critic 구조의 Policy Network.
*   **Architecture & Workflow:**
    1.  **Spatial Encoder:** GATv2Conv와 GraphNorm, 그리고 Residual 연결을 활용하여 맵 구조(Topology)의 로컬 피처를 추출합니다. 입력 노드 차원(`node_dim`)은 `[is_curr, is_tgt, zone_info, dist, is_visited]` 형태의 5차원입니다.
    2.  **Context Generator:** 인코딩된 그래프 출력 텐서 `h`에서 `is_curr`가 1인 노드(현재 에이전트 위치)의 임베딩(`curr_emb`)을 추출합니다. 이 `curr_emb`를 `temporal_proj`에 통과시켜 현재 위치에 대한 고수준 컨텍스트 `h_t`를 생성합니다.
    3.  **Context Broadcasting:** 계산된 현재 위치 컨텍스트 `h_t`를 미니배치 내의 모든 노드에 뿌려줍니다(`h_t.expand` 또는 `h_t[batch]`).
    4.  **Policy Scoring:** 각 노드의 GNN 피처와 방금 뿌려진 `h_t`를 Concat(`torch.cat([h, temporal_out], dim=-1)`)하여 `scorer`와 `value_head`에 입력합니다. 이를 통해 각 노드가 독립적인 피처만 가지는 것이 아니라 "현재 에이전트의 위치 대비 내가 얼마나 좋은 노드인지"를 알 수 있게 됩니다.
    5.  **Masking:** 환경에서 전달받은 `neighbors_mask`를 기반으로 물리적으로 이동 불가능한 노드의 로짓을 `-inf`로 클리핑하여 유효한 액션만 샘플링하도록 유도합니다.

## 2. Tensor & Data Flow
*   `x`: `[N, 5]`
*   `h`: `[N, hidden_dim]` (GNN Spatial Encoder Output)
*   `curr_emb`: `[B, hidden_dim]` (is_curr == True 인 노드들의 임베딩)
*   `h_t`: `[B, hidden_dim]` (Temporal / Context Projection)
*   `temporal_out`: `[N, hidden_dim]` (각 노드에 맞게 Broadcasting된 `h_t`)
*   `combined`: `[N, hidden_dim * 2]` (Actor/Critic 입력 피처)
*   `action_probs`: `[N]` (Softmax 통과 후의 확률 분포)
*   `value`: `[B]` (해당 그래프의 상태 가치, `global_mean_pool` 활용)

## 3. Trial & Error (Debugging History)

### [2026-06-15] 🐛 디버깅: Worker Context Loss (치명적 구조적 버그) 해결
*   **발생 이슈 (Logical Bug):** 코드 리뷰 및 깊은 디버깅(`/code_review`, `/debugging`) 수행 결과, 기존 코드에서 기껏 현재 노드의 임베딩을 추출해놓고(`h_t`), 정작 Policy Head에 합쳐질 때는 **`h`와 `temporal_proj(h)`를 결합해버리는 치명적 실수**가 발견되었습니다.
*   **영향도 분석:** 이 버그로 인해 그래프 상의 노드들은 "현재 에이전트가 어디 있는지"에 대한 GNN 상위 계층의 정보를 전달받지 못하고, 오직 자기 자신의 로컬 피처만으로 평가를 받아 학습 효율과 길찾기 지능이 크게 떨어지는 상태였습니다. (is_curr라는 0/1 플래그에만 의존하는 상태)
*   **해결 및 수정:** `temporal_out` 변수에 들어가는 값을 `temporal_proj(h)`에서 `h_t[batch]` (혹은 배치 없으면 `expand()`)로 변경하여, **"현재 노드"의 문맥 정보를 그래프 전체 노드에 완벽히 Broadcasting 하도록 수정**했습니다. 이제 Worker는 진정한 공간적 관계를 이해하며 월등히 뛰어난 성능을 보일 것입니다.
