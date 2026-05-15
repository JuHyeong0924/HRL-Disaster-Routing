# HRL-Disaster-Routing: 하위 레벨 명세서 (Low-Level Design)

본 문서는 각 모듈의 **함수 시그니처, 인자 타입, 텐서 형태, 내부 로직**을 상세 기술합니다.
전체 아키텍처 개요는 `project_specification_hld.md`(상위 레벨 명세서)를 참조하세요.

---

## 1. 환경 (Environments)

### 1.1. `HRLZoneEnv` (`src/envs/hrl_env.py`)
Phase 1 전용. 재난 없는 정적 맵에서 Zone 기반 길찾기 검증.

#### `__init__(node_file: str, net_file: str, zone_json: str, zone_graph_json: str)`
- `self.num_nodes`: 416 (Anaheim)
- `self.k`: 30 (Zone 개수)
- `self.n2z`: `Dict[int, int]` — 노드 ID → Zone ID
- `self.z2n`: `Dict[int, List[int]]` — Zone ID → 소속 노드 리스트
- `self.ZG`: `nx.Graph` — Zone 인접 그래프 (30 노드)
- `self.hop_matrix`: `np.ndarray [416, 416]` — All-Pairs 홉 거리
- `self._node_zone_tensor`: `torch.Tensor [416]` — 노드별 Zone ID (정적)
- `self._adj_list`: `List[List[int]]` — 인접 리스트 (idx 기반)

#### `reset(batch_size: int = 1) -> torch.Tensor`
- 출력: `[B, 416, 4]` 텐서
- 내부 동작:
  1. B개의 무작위 시종착점 선택 (서로 다른 Zone)
  2. A* 알고리즘으로 Zone Sequence 생성 (예: `[Z3, Z12, Z25]`)
  3. `_get_state_batch()` 호출

#### `_get_state_batch() -> torch.Tensor`
- 출력: `[B, 416, 4]`
- 채널 구성:
  - `[b, :, 0]`: `is_curr` — 현재 노드 one-hot
  - `[b, :, 1]`: `is_tgt` — 최종 목적지 one-hot
  - `[b, :, 2]`: `is_next_zone` — 다음 목표 Zone 소속 노드 마스크
  - `[b, :, 3]`: `hop_dist` — 목적지까지 정규화 홉 거리 (`clamp(100) / 25`)

#### `get_action_mask_batch() -> torch.Tensor`
- 출력: `[B, 416]` (float32, 0.0 또는 1.0)
- 로직: 현재 노드의 이웃 중 `{current_zone, next_zone}`에 속한 것만 `1.0`
- Fallback: 갈 곳이 없으면 자기 자신을 `1.0` (Stagnation 방지)

#### `step_batch(actions: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, List[dict]]`
- 입력: `actions [B]` (노드 인덱스)
- 출력: `state [B, 416, 4]`, `rewards [B]`, `dones [B]`, `infos [B]`
- 보상 체계:
  - `GOAL_REWARD`: +50.0 (목적지 도달)
  - `STEP_PENALTY`: -0.1 (매 스텝)
  - `INVALID_PENALTY`: -10.0 (구역 이탈/제자리)
- Sliding Window: `action_zone == next_zone` → `seq_idx += 1`

---

### 1.2. `HRLClosedLoopEnv` (`src/envs/hrl_closed_loop_env.py`)
Manager-Worker Closed-loop 상호작용 환경 (Phase 2 전용).

#### `__init__(node_file, net_file, worker, k_hop=5, c_max=8, max_manager_turns=50, goal_bonus=10.0, step_penalty_scale=0.1, device='cpu')`
- `self.hop_matrix`: `np.ndarray [N, N]` — APSP 홉 거리
- `self._adj_list`: `List[List[int]]` — 인접 리스트 (Worker 이동용)
- `self.degree_tensor`: `Tensor [N]` — 정규화된 노드 차수

#### `reset() -> Tuple[int, int]`
- 출력: `(current_idx, goal_idx)` — 랜덤 OD쌍

#### `get_manager_state() -> Tensor [N, 4]`
- 채널: `is_curr, is_tgt, hop_dist(목적지 기준), degree`

#### `get_candidate_mask() -> Tensor [N]`
- K-hop 반경 내 (1 ≤ hop ≤ K) 노드만 1.0

#### `_get_worker_state(subgoal_idx: int) -> Tensor [N, 4]`
- 채널: `is_curr, is_tgt, is_subgoal, hop_dist(서브골 기준)`

#### `_get_worker_action_mask() -> Tensor [N]`
- 물리적 인접 이웃 노드만 1.0

#### `execute_worker(subgoal_idx: int) -> Tuple[int, int, bool]`
- Worker가 서브골 방향으로 `c_max` 스텝 이내에 이동 (Greedy 행동)
- 출력: `end_idx`, `steps_taken`, `reached_goal`

#### `step(subgoal_idx: int) -> Tuple[float, bool, Dict]`
- PBRS 보상: `Φ(end) - Φ(start) - 0.1 × steps`, Φ(s) = -hop(s, goal)
- Goal bonus: 최종 목적지 도달 시 +10.0

---

## 2. 모델 (Models)

### 2.1. `Worker` (`src/models/worker.py`)
4-Dim 입력, GATv2 기반 경량 Actor-Critic.

#### `__init__(node_dim: int = 4, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2)`
- `self.convs`: `nn.ModuleList[GATv2Conv]` × `num_layers`
  - 각 레이어: `heads=4, concat=False`
- `self.graph_norms`: `nn.ModuleList[GraphNorm]` × `num_layers`
- `self.input_proj`: `nn.Linear(node_dim, hidden_dim)` — 잔차 연결용
- `self.temporal_proj`: `nn.Sequential(Linear, ReLU)` — LSTM 대체
- `self.scorer`: `nn.Sequential(Linear(hidden_dim*2, hidden_dim), ReLU, Dropout, Linear(hidden_dim, 1))` — 정책 로짓 산출
- `self.critic`: `nn.Sequential(Linear(hidden_dim, hidden_dim), ReLU, Linear(hidden_dim, 1))` — 가치 함수

#### `forward(x, edge_index, batch, neighbors_mask, detach_spatial) -> Tuple[Tensor, Tensor, Tensor]`
- 입력:
  - `x`: `[N, 4]` (또는 배치 시 `[B*N, 4]`)
  - `edge_index`: `[2, E]`
  - `batch`: `[N]` 또는 `None` (단일 그래프)
  - `neighbors_mask`: `[N]` (0.0/1.0)
- 출력:
  - `probs`: `[N]` — 마스킹된 softmax 확률
  - `value`: `[Batch, 1]` — 상태 가치
  - `h_t`: `[Batch, hidden_dim]` — 현재 노드 임베딩

### 2.2. `ReactiveManager` (`src/models/reactive_manager.py`)
비자기회귀 단일 서브골 예측 Manager. Worker와 통일된 GATv2 + Dual Head 구조.

#### `__init__(node_dim: int = 4, hidden_dim: int = 256, num_layers: int = 2, gat_heads: int = 4, dropout: float = 0.2)`
- `self.convs`: `nn.ModuleList[GATv2Conv]` × `num_layers` (heads=4, concat=False)
- `self.graph_norms`: `nn.ModuleList[GraphNorm]` × `num_layers`
- `self.input_proj`: `nn.Linear(node_dim, hidden_dim)` — Residual Connection
- `self.actor`: `nn.Sequential(Linear(hidden_dim*3, hidden_dim), ReLU, Dropout, Linear(hidden_dim, 1))` — 서브골 점수
- `self.critic`: `nn.Sequential(Linear(hidden_dim*2, hidden_dim), ReLU, Linear(hidden_dim, 1))` — 상태 가치 V(s)

#### `forward(x, edge_index, current_idx, goal_idx, candidate_mask, batch) -> Tuple[probs, value, logits]`
- 입력:
  - `x`: `[N, 4]` (is_curr, is_tgt, hop_dist, degree)
  - `candidate_mask`: `[N]` (K-hop 내 후보 1.0, 나머지 0.0)
- 출력:
  - `probs`: `[N]` — K-hop 마스킹된 서브골 선택 확률
  - `value`: `[1]` — Critic 상태 가치
  - `logits`: `[N]` — Raw logits

#### `select_action(x, edge_index, current_idx, goal_idx, candidate_mask, batch, deterministic) -> Tuple[int, Tensor, Tensor, Tensor]`
- PPO 학습/평가를 위해 서브골 1개를 선택 (Categorical 샘플링 또는 argmax)
- 출력: `action (int)`, `log_prob [1]`, `value [1]`, `entropy [1]`

---

## 3. 트레이너 (Trainers)

### 3.1. `HRLWorkerTrainer` (`src/trainers/worker_trainer.py`)
Phase 1 Worker 전용. Gradient Accumulation 기반 REINFORCE.

#### `__init__(env, manager, worker, config)`
- `self.accum_batch`: `config.num_pomo` (기본 16) — K개 에피소드 gradient 누적
- **Ablation 플래그**:
  - `self.use_gae`: GAE(λ) 적용 여부 (기본 `False`)
  - `self.entropy_coeff`: Entropy Bonus 계수 (기본 `0.0`)
  - `self.use_cosine_lr`: Cosine LR Scheduler (기본 `False`)

#### `_run_batch_episodes(batch_size: int) -> list`
- `batch_size` 묶음을 한 번에 GNN Forward 처리
- 각 에피소드의 loss, reward, success, path_len 계산 후 리스트로 반환

#### `train(episodes: int) -> None`
- K개 에피소드 단위로 `_run_batch_episodes()` 실행
- 결과 loss들의 평균으로 `backward()` 및 `optimizer.step()` 호출
- Best/Final 체크포인트 + `runtime_config.json` 자동 저장

### 3.2. `ManagerPPOTrainer` (`src/trainers/manager_ppo_trainer.py`)
Manager v2 전용. PPO + GAE 기반 순수 RL 학습기.

#### `RolloutBuffer`
- 경험 저장: `states`, `actions`, `rewards`, `values`, `log_probs`, `dones`
- `compute_gae(gamma, lam)`: 에피소드 데이터를 역순으로 순회하며 GAE Advantage 및 Return 계산

#### `__init__(env: HRLClosedLoopEnv, manager: ReactiveManager, config)`
- PPO 파라미터: lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_epochs=4

#### `collect_rollouts() -> Dict`
- N개 에피소드 실행 → Buffer에 경험 저장
- Manager가 서브골 선택 후 HRLClosedLoopEnv에서 step 진행 (Worker 이동 처리)
- 에피소드 종료 후 `buffer.compute_gae()` 호출

#### `update() -> Dict`
- `n_epochs` 만큼 버퍼 데이터를 순회
- PPO Clipped Objective 계산: `ratio = exp(new_log_prob - old_log_prob)`
- Critic Loss: `MSE(V_pred, target_return)`
- 전체 Loss: `actor_loss + value_coeff * critic_loss - entropy_coeff * entropy`

#### `train(episodes: int) -> None`
- 매 iteration: collect_rollouts() → update() → 로깅
- Best/Final 체크포인트 + Learning Curve 자동 저장

---

## 4. 에이전트 및 유틸리티 모듈

### 4.1. `src/agents/robot.py`
시뮬레이션 환경 내 물리적 특성을 가진 로봇 에이전트 모듈 (향후 통합).

- **`BaseRobot`**: RoboCue-X 컨셉 모델 (배터리, 속도, 험지 주행 패널티 처리 로직)
  - `_calculate_physics(length, status, damage)`: HAZUS 도로 손상도 기반 주행 속도/에너지 소모량 도출
  - `move(map_instance, dt)`: dt 단위 시간만큼 물리 시뮬레이션 이동 처리
- **`UGV`**: BaseRobot 상속

### 4.2. `src/utils/types.py`
- **`Task`**: 임무 객체 (정찰, 구조, 보급 등), 목적지 노드/좌표 및 필요 자원 정보
- **`AgentState`**: 로봇의 실시간 상태 정보 (잔여 배터리, 현재 이동 경로, 상태)

### 4.3. `src/data/generate_expert.py` & `segment_loader.py`
- 전문가 경로 생성 및 세그먼트 데이터 로딩. (주로 레거시 SL 학습 시 사용됨)

---

## 5. 학습 파이프라인 진입점

### `train_rl.py`
```python
# --stage별 라우팅
if stage == "manager_v2":
    # Phase 2: ReactiveManager + PPO + PBRS Closed-loop
    reactive_mgr = ReactiveManager(node_dim=4, hidden_dim=256).to(device)
    cl_env = HRLClosedLoopEnv(node_file, net_file, worker, k_hop=5, c_max=8)
    trainer = ManagerPPOTrainer(cl_env, reactive_mgr, config)
elif stage == "worker":
    # Phase 1: HRLZoneEnv (Worker 단독 RL 학습)
    env = HRLZoneEnv(...)           
    trainer = HRLWorkerTrainer(env, manager, worker, config)
elif stage == "phase1":
    # Phase 1 Pipeline: Worker -> Manager -> Alignment 자동 순차 실행
    pass
elif stage == "phase1_parallel":
    # Multi-GPU 병렬 Phase 1 실행
    pass
```

### 주요 CLI 인자
| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--stage` | `phase1` | `worker`, `manager_v2`, `manager`, `alignment`, `phase1`, `phase1_parallel` |
| `--episodes` | 5000 | 학습 에피소드 수 |
| `--batch_size` | 16 | Gradient Accumulation (Worker) 또는 Rollout 수 (Manager v2) |
| `--hidden_dim` | 256 | 모델 히든 차원 |
| `--lr` | 1e-4 | 학습률 |
| `--num_layers` | 2 | Worker/Manager GATv2 레이어 개수 |
