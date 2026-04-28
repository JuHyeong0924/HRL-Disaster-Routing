# HRL-Disaster-Routing: 하위 레벨 명세서 (Low-Level Design)

본 문서는 각 모듈의 **함수 시그니처, 인자 타입, 텐서 형태, 내부 로직**을 상세 기술합니다.
전체 아키텍처 개요는 `@project_specification.md`(상위 레벨 명세서)를 참조하세요.

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

## 2. 모델 (Models)

### 2.1. `Worker` (`src/models/worker.py`)
4-Dim 입력, GATv2 기반 경량 Actor-Critic.

#### `__init__(node_dim: int = 4, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2)`
- `self.convs`: `nn.ModuleList[GATv2Conv]` × `num_layers`
  - 각 레이어: `heads=4, concat=False`
- `self.graph_norms`: `nn.ModuleList[GraphNorm]` × `num_layers`
- `self.input_proj`: `nn.Linear(node_dim, hidden_dim)` — 잔차 연결용
- `self.temporal_proj`: `nn.Linear(hidden_dim, hidden_dim)` — LSTM 대체
- `self.scorer`: `nn.Linear(hidden_dim * 2, 1)` — 정책 로짓 산출
- `self.critic`: `nn.Linear(hidden_dim, 1)` — 가치 함수

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
- 내부 흐름:
  1. `_forward_gnn(x, edge_index, batch)` → `h [N, 256]`
  2. `curr_emb = h[x[:, 0].bool()]` → `[Batch, 256]`
  3. `h_t = temporal_proj(curr_emb)` → `[Batch, 256]`
  4. `scorer_input = cat([h, h_t_expanded], dim=-1)` → `[N, 512]`
  5. `logits = scorer(scorer_input).squeeze(-1)` → `[N]`
  6. `logits.masked_fill(mask == 0, -inf)` → Softmax → `probs [N]`

### 2.2. `GraphTransformerManager` (`src/models/node_manager.py`)
Node-level Pointer Network 기반 Manager. Phase 1에서는 미사용(Dummy A*).

### 2.3. `ZoneManager` (`src/models/zone_manager.py`)
Zone-level GCN Manager. Phase 2에서 구현 예정 (스켈레톤).

---

## 3. 트레이너 (Trainers)

### 3.1. `HRLWorkerTrainer` (`src/trainers/worker_trainer.py`)
Phase 1 Worker 전용. Gradient Accumulation 기반 REINFORCE.

#### `__init__(env, manager, worker, config)`
- `self.accum_batch`: `config.num_pomo` (기본 16) — K개 에피소드 gradient 누적
- `self.edge_index`: `[2, E]` (정적, GPU 캐싱)
- Manager는 `eval()` + `requires_grad_(False)` 동결

#### `_run_single_episode() -> dict`
- 단일 에피소드 실행 → `{'loss', 'reward', 'success', 'path_len'}` 반환
- Loss 계산: `policy_loss + value_loss` (backward 미호출)

#### `train(episodes: int) -> None`
- 매 에피소드마다 `_run_single_episode()` 실행
- `loss / K`로 스케일링 후 `backward()` 호출 (gradient 누적)
- K개마다 `optimizer.step()` + `zero_grad()`
- Best/Final 체크포인트 자동 저장

### 3.2. `WorkerNavTrainer` (`src/trainers/worker_nav_trainer.py`)
레거시 7-Dim Worker 학습용. POMO + PBRS + Hidden Checkpoint.

### 3.3. `DOMOTrainer` (`src/trainers/pomo_trainer.py`)
Joint(Alignment) 학습. Manager-Worker 동시 최적화. POMO 64 배치.

### 3.4. `ManagerStageTrainer` (`src/trainers/manager_stage_trainer.py`)
Manager 단독 RL 학습. Worker 동결 상태.

---

## 4. 학습 파이프라인 진입점

### `train_rl.py`
```python
# --stage별 라우팅
if stage == "worker":
    env = HRLZoneEnv(...)           # Phase 1 환경
    trainer = HRLWorkerTrainer(env, manager, worker, config)
elif stage == "manager":
    env = DisasterEnv(...)          # 재난 환경
    trainer = ManagerStageTrainer(env, manager, worker, config)
elif stage == "alignment":
    env = DisasterEnv(...)
    trainer = DOMOTrainer(env, manager, worker, config)
```

### 주요 CLI 인자
| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--stage` | `phase1` | `worker`, `manager`, `alignment`, `phase1`, `phase1_parallel` |
| `--episodes` | 5000 | 학습 에피소드 수 |
| `--num_pomo` | auto | Gradient Accumulation 배치 크기 |
| `--hidden_dim` | 256 | 모델 히든 차원 |
| `--lr` | 1e-4 | 학습률 |
