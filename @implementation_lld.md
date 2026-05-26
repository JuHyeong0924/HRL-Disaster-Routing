# HRL-Disaster-Routing: 하위 레벨 명세서 (Low-Level Design)

본 문서는 각 모듈의 **함수 시그니처, 인자 타입, 텐서 형태, 내부 로직 및 세부 수학 공식**을 상세 기술합니다.
전체 아키텍처 개요는 `@architecture_hld.md`(상위 레벨 명세서)를 참조하세요.

---

## 1. 환경 (Environments)

### 1.1. `HRLZoneEnv` (`src/envs/hrl_env.py`)
Phase 1 전용. 재난이 배제된 정적 맵에서 Zone 가이드 또는 Node 가이드 기반 길찾기 성능을 병렬(POMO 배치)로 검증합니다.

#### 생성자 `__init__(node_file: str, net_file: str, zone_json: str = '...', zone_graph_json: str = '...', masking_mode: str = 'hard', use_pbrs: bool = False, subgoal_mode: str = 'zone')`
*   `self.num_nodes`: 416 (Anaheim 기준)
*   `self.max_hop`: 그래프의 실제 maximum diameter 홉 수 (정규화 분모로 사용)
*   `self.n2z`: `Dict[int, int]` — 노드 ID → Zone ID
*   `self.z2n`: `Dict[int, List[int]]` — Zone ID → 소속 노드 ID 리스트
*   `self.ZG`: `nx.Graph` — Zone 인접 그래프 (30 노드)
*   `self.hop_matrix`: `np.ndarray [416, 416]` — BFS 탐색으로 계산 및 디스크에 자동 캐싱 (`data/hop_matrix_*.npy`)
*   `self.masking_mode`: `'hard' | 'hard_full_seq' | 'soft_curr_next' | 'soft_flex'`
*   `self.use_pbrs`: True일 경우 홉 거리 차이에 의한 dense potential reward 활성화
*   `self.subgoal_mode`: `'zone' | 'node'` (가이드 대상 분기)

#### `reset(batch_size: int = 1) -> torch.Tensor`
*   **출력:** `state` 텐서 `[B, 416, 4]` (float32)
*   **내부 동작:**
    1. B개의 독립적인 무작위 시종착점(s, t) 선택 (서로 다른 Zone)
    2. A* 알고리즘으로 Zone Sequence 생성 (Zone 모드) 및 최단 경로 Node Sequence 생성 (Node 모드)
    3. Node 모드 시, 시퀀스 상 3-hop 또는 6-hop 앞의 노드를 초기 `subgoal_nodes[b]`로 바인딩
    4. `_get_state_batch()` 호출 및 반환

#### `_get_state_batch() -> torch.Tensor`
*   **출력:** `state` 텐서 `[B, 416, 4]`
*   **채널 구성:**
    *   `[b, :, 0]`: `is_curr` — 현재 노드 핫 코딩 `[B, 416]` (1.0 or 0.0)
    *   `[b, :, 1]`: `is_tgt` — 최종 목적지 핫 코딩 `[B, 416]`
    *   `[b, :, 2]`: `is_subgoal` / `is_next_zone` — `subgoal_mode`가 `'zone'`일 경우 다음 목표 Zone 소속 노드 마스크, `'node'`일 경우 특정 서브골 노드 위치만 `1.0`
    *   `[b, :, 3]`: `hop_dist` — 목적지까지의 홉 거리 정규화 값 (`torch.clamp(hops, max=100.0) / max(self.max_hop, 1.0)`)

#### `get_action_mask_batch() -> torch.Tensor`
*   **출력:** `mask` 텐서 `[B, 416]` (float32)
*   **내부 로직:**
    *   `hard`: 현재 노드의 물리적 이웃 중 `{current_zone, next_zone}`에 포함된 노드만 `1.0`
    *   `hard_full_seq`: 현재 노드의 물리적 이웃 중 `zone_sequences[b]` 전체에 포함된 노드만 `1.0`
    *   `soft_curr_next` / `soft_flex`: 물리적 제약 조건을 해제하여 모든 이웃 노드를 `1.0`으로 허용 (구역 이탈은 페널티로 유도)
    *   **Fallback:** 갈 수 있는 노드가 전혀 존재하지 않는 경우, 제자리 머무름을 방지하기 위해 자기 자신 노드를 `1.0`으로 강제 허용.

#### `step_batch(actions: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, List[dict]]`
*   **입력:** `actions` 텐서 `[B]` (샘플링된 노드 인덱스)
*   **출력:** `next_state [B, 416, 4]`, `rewards [B]`, `dones [B]`, `infos [List[dict]]`
*   **보상 공식:**
    *   물리적 비인접 노드 선택 or 제자리 선택: `rewards[b] = INVALID_PENALTY = -10.0` 및 에피소드 강제 종료 (`dones[b] = True`)
    *   최종 목적지 도달: `rewards[b] = GOAL_REWARD = +50.0`
    *   스텝 페널티: `STEP_PENALTY = -0.1`
    *   OOB 페널티 (soft 모드 한정, 지정된 Zone을 벗어난 이웃 이동 시): `rewards[b] += OOB_PENALTY = -0.5`
    *   **Sliding Window:** `action_zone == next_zone` 진입 시 `zone_seq_idxs[b] += 1` 처리.
    *   **[P0 중간보상]** `zone_progress_reward = True`일 경우 Zone 전환 성공 시 진행률에 따른 중간보상 (`5.0 * (progress)`) 부여.
    *   **[v3 PBRS]** `use_pbrs = True`일 경우:
        $$\text{PBRS} = (\text{prev\_hop} - \text{new\_hop}) \times 0.5$$
        (목적지 또는 서브골 노드에 가까워지면 양수 보상, 멀어지면 음수 페널티를 `rewards[b]`에 즉시 합산)

---

### 1.2. `HRLClosedLoopEnv` (`src/envs/hrl_closed_loop_env.py`)
Manager-Worker Closed-loop 상호작용 환경 (Phase 2 전용).

#### 생성자 `__init__(node_file: str, net_file: str, worker: torch.nn.Module, k_hop: int = 5, c_max: int = 8, max_manager_turns: int = 50, goal_bonus: float = 10.0, step_penalty_scale: float = 0.1, device: str = 'cpu') -> None`
*   `self.edge_index`: TNTP 로드 후 GNN 학습用に 구축한 정적 엣지 인덱스 `[2, E]`
*   `self.edge_attr`: `[E, 3]` — Min-Max 정규화가 적용된 `[length, capacity, speed]` 도로 특성 텐서

#### `reset() -> Tuple[int, int]`
*   **출력:** `(current_idx, goal_idx)` — 무작위로 선택된 도달 가능한 출발지-목적지 노드 쌍

#### `get_manager_state() -> torch.Tensor`
*   **출력:** `x` 텐서 `[N, 4]` (is_curr, is_tgt, hop_dist, degree)
*   **채널:**
    *   `x[current_idx, 0] = 1.0` (현재 위치)
    *   `x[goal_idx, 1] = 1.0` (최종 목적지)
    *   `x[:, 2]`: 최종 목적지 기준 홉 거리를 `self.max_hop`으로 나눈 정규화 텐서
    *   `x[:, 3]`: 노드의 NetworkX 차수(degree)를 최대 차수로 나눈 정규화 텐서

#### `get_candidate_mask() -> torch.Tensor`
*   **출력:** `mask` 텐서 `[N]` (float32)
*   **로직:** 현재 노드와의 홉 거리(hop)가 $1 \le \text{hop} \le K$ 인 모든 노드를 `1.0`으로 활성화.
*   **예외:** 최종 목적지 노드가 K-hop 이내에 들어와 있다면, 우회 없는 직접 도달을 유도하기 위해 목적지 노드를 마스크에 강제로 포함(`mask[goal_idx] = 1.0`).

#### `_get_worker_state(subgoal_idx: int) -> torch.Tensor`
*   **출력:** `x` 텐서 `[N, 4]` (is_curr, is_tgt, is_subgoal, hop_dist_to_subgoal)

#### `execute_worker(subgoal_idx: int) -> Tuple[int, int, bool]`
*   **내부 동작:** Worker가 서브골을 향해 탐욕적(Greedy)으로 이동하며, 최대 `c_max` 스텝만큼 전진을 허용합니다.
*   **출력:** `(end_idx, steps_taken, reached_goal)`

#### `step(subgoal_idx: int) -> Tuple[float, bool, Dict]`
*   **수식 기반 PBRS 보상 산출:**
    $$\text{Potential} = \Phi(s) = -\text{hop\_dist}(s, \text{goal})$$
    $$\text{Reward} = [\Phi(\text{end\_idx}) - \Phi(\text{start\_idx})] - (\text{step\_penalty\_scale} \times \text{steps\_taken})$$
*   최종 목적지 도착 시 `goal_bonus` (+10.0) 추가 합산.

---

## 2. 모델 (Models)

### 2.1. `Worker` (`src/models/worker.py`)
2-Layer GATv2 기반 경량 Actor-Critic 신경망.

#### 생성자 `__init__(node_dim: int = 4, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2, use_checkpoint: bool = False, use_jk_net: bool = False, use_edge_attr: bool = False)`
*   `use_checkpoint`: VRAM이 부족한 훈련 국면에서 Gradient Checkpointing(`grad_ckpt`)을 활성화하기 위한 플래그
*   `use_jk_net`: Jumping Knowledge Net 적용 여부 (활성화 시 레이어별 은닉 임베딩을 concatenate 한 뒤 `self.jk_proj`로 차원 축소)
*   `use_edge_attr`: GATv2Conv 메시지 패싱 시 도로 엣지 속성(`[length, capacity, speed]`) 반영 여부

#### `forward(x, edge_index, batch, neighbors_mask=None, detach_spatial=False, edge_attr=None) -> Tuple[Tensor, Tensor, Tensor]`
*   **입력 텐서 형상:**
    *   `x`: `[N_total, 4]` (배치 노드 피처)
    *   `edge_index`: `[2, E_total]`
    *   `batch`: `[N_total]` (그래프 배치 인덱스)
    *   `neighbors_mask`: `[N_total]` (행동 제한 마스크)
    *   `detach_spatial`: 공간(GNN) 그래디언트 흐름 차단 여부
*   **출력 텐서 형상:**
    *   `probs`: `[N_total]` — 마스킹 처리된 Softmax 정책 확률 분포
    *   `value`: `[Batch, 1]` — Critic 상태 가치 추정값 V(s)
    *   `h_t`: `[Batch, hidden_dim]` — Temporal(Linear) 투영이 반영된 현재 노드 임베딩

---

### 2.2. `ReactiveManager` (`src/models/reactive_manager.py`)
비자기회귀 단일 서브골 예측 Manager. Transformer Decoder를 배제하고 MLP Scorer 방식을 채택했습니다.

#### 생성자 `__init__(node_dim: int = 4, hidden_dim: int = 256, num_layers: int = 2, gat_heads: int = 4, dropout: float = 0.2)`
*   `self.actor`: 3배 확장 임베딩 점수 산출기
    *   `Linear(hidden_dim * 3, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)`
*   `self.critic`: 가치 함수 예측기
    *   `Linear(hidden_dim * 2, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)`

#### `forward(x, edge_index, current_idx, goal_idx, candidate_mask, batch=None) -> Tuple[Tensor, Tensor, Tensor]`
*   **내부 동작:**
    1. GNN 인코더를 거쳐 전체 그래프 노드 임베딩 $h$ `[N, hidden_dim]`를 추출
    2. 현재 노드 임베딩 $h_{\text{curr}}$와 목적지 노드 임베딩 $h_{\text{goal}}$을 추출
    3. 전역 노드 $h$를 순회하며 Actor 입력 벡터 형성:
       $$\text{Input}_{\text{actor}} = [h_{\text{curr}} \parallel h_{\text{goal}} \parallel h]$$ (차원: `[N, 3 * hidden_dim]`)
    4. Scorer를 통과시켜 `logits`를 산출하고, 후보군이 아닌 노드는 `float('-inf')`로 하드 마스킹 처리 후 `F.softmax` 적용
*   **출력 텐서 형상:**
    *   `probs`: `[N]` (서브골 선택 확률)
    *   `value`: `[1]` (Critic 상태 가치)
    *   `logits`: `[N]` (마스킹 적용 전 raw logits)

---

## 3. 트레이너 (Trainers)

### 3.1. `HRLWorkerTrainer` (`src/trainers/worker_trainer.py`)
POMO 배치 병렬 및 Gradient Accumulation 기반 REINFORCE 트레이너.

#### `_compute_gae(rewards: list, values: list) -> torch.Tensor`
*   **GAE 계산 로직:** 역순 순회를 통한 Advantage 산출
    $$\delta_t = R_t + \gamma V_{t+1} - V_t$$
    $$A_t^{\text{GAE}} = \delta_t + \gamma \lambda A_{t+1}^{\text{GAE}}$$

#### `_run_batch_episodes(batch_size: int) -> list`
*   **배치 컴팩팅 기법:** 미종료 에피소드만 골라 GNN 연산을 한 번에 처리 (`active` 리스트 및 `to_dense_batch` 형태의 슬라이싱 수행)하여 CPU-GPU 병목 현상을 원천 차단합니다.

#### `train(episodes: int) -> None`
*   매 step마다 `num_pomo` 개수만큼의 에피소드를 병렬로 실행하고 gradient를 누적하여 업데이트.
*   Cosine Annealing LR (`scheduler.step()`) 연동.
*   `logs/rl_worker_stage/<timestamp>/` 디렉토리에 `best.pt`, `final.pt` 가중치 저장 및 하이퍼파라미터 정보가 담긴 `runtime_config.json`을 저장.

---

### 3.2. `ManagerPPOTrainer` (`src/trainers/manager_ppo_trainer.py`)
Manager v2를 최적화하는 PPO + GAE 기반 순수 강화학습 트레이너.

#### `RolloutBuffer.compute_gae(gamma: float, lam: float) -> None`
*   에피소드 역순 Advantage 계산 후, Gradient Exploding 방지를 위해 Z-score 정규화 수행:
    $$A_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}$$

#### `update() -> Dict`
*   `n_epochs` 동안 데이터를 순회하며 PPO Clipped Loss를 계산:
    $$r_t(\theta) = \exp(\log \pi_{\theta}(a_t|s_t) - \log \pi_{\theta_{\text{old}}}(a_t|s_t))$$
    $$L^{\text{CLIP}}(\theta) = -\mathbb{E}\left[ \min(r_t(\theta)A_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) \right]$$
    $$L^{\text{Critic}}(\phi) = \text{MSE}(V_{\phi}(s_t), G_t)$$
    $$L^{\text{Total}} = L^{\text{CLIP}} + c_1 L^{\text{Critic}} - c_2 \mathcal{H}(\pi_{\theta})$$

#### `train(episodes: int) -> None`
*   최종 학습 완료 시 `learning_curve.png` 곡선 시각화 그래프를 자동 렌더링하여 `logs/rl_manager_v2/<run_label>/`에 생성 보관.

---

## 4. 에이전트 및 유틸리티 모듈

### 4.1. `src/agents/robot.py` (RoboCue-X 물리 모델)
전고체 배터리와 고마력 트랙션을 갖춘 고기동 인텍 수송 크롤러 물리 엔진.

*   **스펙 요약:** `mass = 200kg`, `battery = 10kWh` ($3.6 \times 10^7$ Joule), `base_speed = 40km/h`, 평지 기준 전비 `efficiency = 50 Wh/km`

#### 주행 부하 계산식 (`_calculate_physics`):
HAZUS 도로 손상 등급에 맞춰 실시간 주행 상태 도출:
1.  **Normal (Slight Damage $\le 0.2$):** 속도 계수 `traction = 0.9`, 부하 계수 `load = 1.1`
2.  **Caution (Moderate Damage $\le 0.5$):** 속도 계수 `traction = 0.6`, 부하 계수 `load = 1.5`
3.  **Danger (Extensive Damage $\le 0.8$):** 속도 계수 `traction = 0.3`, 부하 계수 `load = 3.0`
4.  **Closed (Complete Damage $> 0.8$):** 속도 계수 `traction = 0.0`, 부하 계수 `load = inf` (주행 불가)

*   **실제 주행 속도:**
    $$v_{\text{real}} = v_{\text{base}} \times \text{traction}$$
*   **요구 전력량:**
    $$\text{Wh}_{\text{req}} = \text{length\_km} \times (\text{rated\_efficiency} \times \text{load})$$
*   **소모 배터리 (Joule):**
    $$E_{\text{joule}} = \text{Wh}_{\text{req}} \times 3600.0$$

---

### 4.2. `src/utils/types.py` (데이터 구조 메타데이터)
```python
@dataclass
class Task:
    task_id: int                    # 임무 고유 식별값
    task_type: str                  # "RECON" | "RESCUE" | "SUPPLY"
    node_id: int                    # 임무 발생 노드 번호
    location: Tuple[float, float]   # (x, y) 절대 좌표
    priority: int                   # 우선순위 (1 ~ 3)
    status: str                     # 상태 ("PENDING", "ASSIGNED", "COMPLETED")
    assigned_agent_id: Optional[str] # 배정된 로봇 ID
    required_resources: Dict[str, float] # 필요 자원 {"work_time": 30.0, "battery_cost": 5.0}

@dataclass
class AgentState:
    agent_id: str                   # 로봇 ID
    agent_type: str                 # "UGV"
    current_node: int               # 현재 머무는 노드 인덱스
    current_edge: Tuple[int, int]   # 이동 중인 엣지 정보
    position: Tuple[float, float]   # (x, y) 실시간 절대 좌표
    battery: float                  # 잔여 배터리 (0.0 ~ 100.0)
    status: str                     # "IDLE" | "MOVING" | "WORKING" | "RESUPPLYING"
    assigned_task_queue: List[int]  # 수행 임무 대기열
    current_path: List[int]         # 현재 주행 예정 경로 노드 리스트
```

---

## 5. 학습 파이프라인 진입점 (`train_rl.py`)

### 주요 CLI 파라미터 및 Preset 시스템

*   `--stage`: 실행 단계 라우팅 (`worker`, `manager_v2`, `manager`, `alignment`, `phase1`, `phase1_parallel`)
*   `--episodes`: 총 에피소드 학습 횟수
*   `--batch_size`: gradient 누적 크기 (Worker) 또는 Rollout 수 (Manager v2)
*   `--masking_mode`: Action Masking 범위 제어 (`hard`, `hard_full_seq`, `soft_curr_next`, `soft_flex`)
*   `--use_pbrs`: Worker PBRS 보상 적용 여부
*   `--use_jk_net` & `--use_edge_attr`: Worker GNN 최적화 레이어 제어 플래그
*   `--mgr_state_preset`: `S0` ~ `S13` 에 따른 관찰 피처 차원 매핑 시스템 (`S7` 기본값 = 4)
*   `--bias_preset`: 디코더 마스크 프리셋 (`full`, `none`, `khop_only`, `soft_only`)
*   `--reward_preset`: PPO 보상 학습 프리셋 (`full`, `minimal`, `mid`, `proximity`)
