# Tier 2: Architecture Design (Mid - Detailed Design)

## 1. System Schemas & Modules

시스템의 코어 모듈들은 `src/` 내에서 각각의 뚜렷한 책임을 가집니다. Python의 Type Hinting과 명확한 API Contract를 준수하여 설계되었습니다.

### 1.1. Environment Modules (`src/envs/`)

#### `disaster_map.py` & `disaster_env.py`
- **역할**: TNTP 기반의 도로망 데이터를 NetworkX 그래프로 로드하고 관리합니다.
- **HAZUS Soft Closure (Phase 1A)**: `apply_disaster_damage()`는 FEMA HAZUS Earthquake Model의 Residual Capacity 기반 가중치 체계를 적용합니다. **간선을 절대 제거하지 않아** 그래프 연결성을 항상 보장합니다.
  - **Damage → Status → Weight Multiplier 매핑**:
    | Damage 범위 | HAZUS 등급 | Residual Capacity | Weight Multiplier | UGV 파괴 확률 |
    |-----------|----------|------------------|------------------|--------------|
    | 0.0~0.2 | Slight (Normal) | 100% | ×1.0 | 0% |
    | 0.2~0.5 | Moderate (Caution) | 50% | ×2.0 | 0% |
    | 0.5~0.8 | Extensive (Danger) | 25% | ×4.0 | 10% (Trap) |
    | 0.8~1.0 | Complete (Closed) | ~5% | ×20.0 | 30% |
  - **Severity Roll 분포**: 40% Slight, 30% Moderate, 25% Extensive, 5% Complete
  - **데미지 누적**: `damage = min(1.0, 기존 + 신규)`. `is_reset` 모드에서 전체 초기화.

#### `worker_env.py` (`WorkerEnv`)
- **역할**: Worker 단독 학습(Phase 1)을 위한 라우팅 환경. 하나의 Zone 내부 혹은 근접 인접 Zone으로 이동하는 세부 노드 스텝을 제어. 실제 물리적 거리(Dijkstra APSP, `weight='weight'`)를 기반으로 한 상태 공간과 보상 체계를 제공.
- **State Schema (Tensor Shape: `[N, 4]` or `[N, 5]`)**: 
  - `is_curr` (Channel 0): 현재 노드 위치 (One-hot, 1.0 or 0.0)
  - `is_tgt` (Channel 1): 최종 목표 노드 위치 (One-hot, 1.0 or 0.0)
  - `zone_info` (Channel 2): Zone 정보를 스칼라 값으로 압축 (`-1`: 금지/이동 외 구역, `0`: 현재 소속 구역, `1`: 다음 이동해야 할 목표 Subgoal 구역)
  - `dist` (Channel 3): 목표까지의 물리적 최단 거리. 학습 안정성을 위해 정규화 (`torch.clamp(dists, max=100000.0) / max(self.max_dist, 1.0)` 또는 `use_relative_hop` 사용 시 상대적 거리 감소폭)
  - `is_visited` (Channel 4, 선택적): 과거 방문한 노드 여부(순환 궤적 방지용, `use_is_visited=True` 일 때 활성화)
- **Action Schema**: 현재 노드와 연결된 직접적인 이웃 노드(Neighbor Node ID). Masking을 통해 연결되지 않은 노드는 Logit = `-inf` 처리.
  - **Neighbor-Scoped Masking Loop (최적화)**: `get_action_mask_batch()` 내에서 전체 노드 $N=416$개를 선형 스캔하는 대신, 초기 인접 노드 리스트 `self._adj_list[c_idx]` 상의 노드들만 순회 검사하여 복잡도를 $O(N)$에서 $O(\text{deg}(v))$로 개선.
- **Reward Schema**: 
  - Sparse Reward (Goal): 최종 타겟 또는 목표 구역 도달 시 강력한 양의 보상(`+50.0`, `+5.0`) 부여.
  - Base Penalty (Time): 소모되는 물리적 거리(Edge weight)에 비례하는 지속적인 타임 패널티(`-0.1 * weight`).
  - Constraint Penalty (Revisit): 탐색 과정의 무한 루프(Stagnation)를 방지하기 위해, 이미 방문했던 노드(`visited_nodes == 1.0`)로 되돌아가려 할 경우 강력한 마이너스 보상(`-5.0`)을 즉각 부여.
  - Dense Reward (PBRS): 목표 노드까지의 물리적 거리를 기반으로 한 Potential-Based Reward Shaping. 거리가 줄어들면 양(+)의 보상, 멀어지면 음(-)의 보상을 주어 탐색 유도 (식: $\Phi(s) = max\_dist - dist\_to\_goal$).
- **Vectorization & Tensor Memory (병목 혁신)**: 기존 NetworkX 기반의 파이썬 Dictionary 룩업(`G[u][v]`, `G.has_edge()`)에 의존하던 환경 시뮬레이션을 전면 폐기하였습니다. 대신 환경 초기화 및 재난 이벤트(Aftershock) 직후 `sync_tensors_from_graph()` 메서드를 호출하여 Graph topology를 GPU 상의 고밀도 매트릭스 텐서군으로 동기화합니다. 
  - `_adj_matrix_tensor` `[N, N]` bool: 연결성 캐싱.
  - `_weight_matrix` `[N, N]` float32: 실시간 물리적 간선 가중치 캐싱.
  - `_damage_matrix` `[N, N]` float32: 파괴 정도(Damage) 캐싱.
  - `_status_matrix` `[N, N]` int8: 도로 상태(0=Normal, 1=Caution, 2=Danger, 3=Closed) 캐싱.
  - `_dist_matrix_tensor` `[N, N]` float32: Dijkstra APSP 최단 경로 거리 행렬.
  이를 통해 환경의 전이(State Transition), 보상, 피처 추출 로직에서 루프 연산을 `O(1)` 행렬 인덱싱으로 대체하여 CPU Starvation 병목을 완전히 해결했습니다.

#### `manager_env.py` (`ManagerEnv`)
- **역할**: Manager 학습(Phase 2)을 위한 비자기회귀(Non-autoregressive) Closed-loop 계층 환경. 매크로 관점에서 전체 Zone Graph를 탐색하여 최적의 다음 Subgoal Zone을 결정하며, 결정 이후엔 내장된 Worker가 실제로 이동함.
- **State Schema (Tensor Shape: `[K, 7]`)**: 
  - `is_curr_zone` (Channel 0): 현재 위치한 Zone (1.0 or 0.0)
  - `is_tgt_zone` (Channel 1): 최종 목적지가 포함된 Zone (1.0 or 0.0)
  - `is_visited_zone` (Channel 2): 에피소드 중 이미 방문했던 Zone (루프 방지)
  - `zone_dist` (Channel 3): 목표 노드에서 해당 구역까지의 최소 물리적 거리 (휴리스틱 $h(n)$ 역할, `max_dist`로 정규화)
  - `distance_from_curr` (Channel 4): 현재 노드에서 해당 구역까지의 최소 물리적 거리 (실제 비용 $g(n)$ 역할, `max_dist`로 정규화)
  - `zone_disaster_intensity` (Channel 5): 해당 구역의 재난 피해도(복잡도/위험도).
  - `cos_sim` (Channel 6): 방향성 특징(Directional Feature). 현재 Zone Centroid에서 목표 Zone Centroid를 향하는 벡터와 후보 Zone 벡터 간의 코사인 유사도.
- **Action Schema**: 
  - 목적지까지 도달 가능한 Zone ID.
  - `masking_mode`에 의해 인접하지 않은 Zone이나 이미 제자리 걸음(Self-loop)을 유도하는 Zone은 Logit = `-inf` 마스킹 처리됨.
- **Reward Schema**: 
  - Base Penalty: Manager 턴 소모에 따른 패널티 (`-0.5`) + Revisit 패널티 (`-5.0`) + Worker의 소모 스텝수에 비례하는 초선형 패널티(`-0.1 * steps^{1.5}`).
  - PBRS: Zone 이동 시 목표와의 물리적 거리가 가까워진 정도에 비례하는 포텐셜 보상. `reward_pbrs = (np.log1p(prev_dist) - np.log1p(curr_dist)) * 2.0` 수식을 사용하여 여진(Aftershock) 등으로 거리가 수만 단위로 폭주하더라도 `11.3` 내외로 압축 정규화하여 가치 신경망의 Loss 발산을 원천 차단함.
- **Vectorization**: `ManagerEnv`의 이너 워커 루프(`step_manager` 내 `for _ in range(50)`) 역시 WorkerEnv의 풀 벡터화 아키텍처를 그대로 상속받으며, CPU-GPU Device 캐스팅 비용을 최소화하기 위해 모든 Zone 추적 텐서(`_node_zone_tensor` 등)를 GPU VRAM에 상주시키고, 인덱싱 연산을 통합하여 실행 속도를 10~50배 이상 비약적으로 향상시켰습니다.

#### `hrl_env.py` (`HRLEnv`)
- **역할**: Manager가 여러 개의 타겟(Multi-OD)을 동시다발적으로 처리하고 데드라인(Deadline)을 관리할 수 있도록 래핑한 최상위 통합 환경.
- **Phase 1B — 시간 기반 Continuous Aftershock**: 여진은 Manager/Worker 턴과 무관하게 `current_time` 축에서 독립적으로 발생 (Omori's Law 근거). `reset()` 시 15~25회 여진 스케줄 생성, 여진 1회당 damage_prob=0.05 적용, Worker 루프 내에서 `aftershock_cursor`로 진행.
- **Phase 1C — UGV 파괴 판정**: Worker step 직후 통과한 간선의 status 확인. Closed 간선 30% 파괴, Danger 간선 10% Trap(시간 페널티 +30.0, 복귀).
- **Phase 2A — Target Features `[B, N, 6]`**:
  | Channel | 의미 | 범위 |
  |---------|------|------|
  | 0 | deadline (정규화) | [0, 1] |
  | 1 | time_remaining (정규화) | [-∞, 1] |
  | 2 | dist_from_curr (정규화) | [0, 1] |
  | 3 | urgency_ratio (dist/rem) | [0, 5] |
  | 4 | feasibility (도달 가능) | {0, 1} |
  | 5 | normalized_slack (여유) | [-1, 1] |
- **Manager-Turn Scoped Memory**: 매 턴마다 Worker의 `visited_nodes`를 초기화하여 단일 임무에만 집중. 이를 통해 워커는 오직 '매니저가 지금 내린 단일 임무'에만 집중하는 이상적인 HRL 철학을 수학적으로 유지합니다.
- **Vectorization & Graph Tensor Memory (최적화)**: 
  * GNN에 필요한 `edge_index` 및 노드 정보(`bidir_nodes`)를 `__init__` 시점에 캐싱하여 `_build_graph_data()` 연산(CPU-GPU 전송 및 NetworkX 구조 조회)을 최소화. 특히 간선 속성(`edge_attr`) 추출 시 기존 Python 룩업을 제거하고 캐싱된 텐서(`_bidir_src`, `_bidir_dst`)를 통한 다차원 슬라이싱(`self.env._damage_matrix[src, dst]`)으로 풀-벡터화 달성.
  * `step_manager()` 내에서 NetworkX 간선 확인(`G.has_edge`)을 GPU 텐서(`_adj_matrix_tensor`, `_status_matrix`)로 완전히 대체.
  * Zone Features, Target Features, Masks 생성 로직(기존 3중 Python for loop)을 PyTorch 브로드캐스팅 및 행렬 곱(`@`)을 통한 $O(1)$ 연산으로 리팩토링. $O(A \times B \times N)$ 제곱배 병목을 완전히 제거.


### 1.2. Simulation Engine & Continuous Time Model (SMDP)
- **시간 흐름(Time Tick) 설계**: 기존의 1-Hop = 1-Step 방식에서 탈피하여, 물리적 가중치 거리(`edge_weight`)에 기반해 실제 소요 시간을 계산하는 **연속 시간(Continuous Time) 모델**로 개편되었습니다.
- **RL 환경 (`hrl_env.py`)**: `current_time`은 단순 스텝 수가 아닌, 워커가 통과하는 도로의 `weight_dist`만큼 더해집니다. 따라서 파괴된 구간을 통과하면 엄청난 시간이 누적되어 타겟 데드라인(TW)을 초과하게 되므로 에이전트가 이를 수학적으로 기피하게 됩니다.
- **시각화 엔진 (`visualize_heuristic.py`)**: `0.5` 간격으로 `global_time` 틱이 발생하며, 워커가 도로에 진입하면 그 가중치에 비례하는 시간 동안 `Busy` 상태가 되어 의사결정이 잠깁니다. 또한, 시간 흐름에 비례하여 출발지와 도착지 간의 좌표를 선형 보간(Linear Interpolation)하여 물리적으로 부드럽게 이동하는 애니메이션을 렌더링합니다.

### 1.3. Neural Network Models (`src/models/`)

#### `worker.py` (`Worker`)
- **Architecture**: Graph Attention Network v2 (GATv2) 중심의 Local-feature Extractor + Actor-Critic MLP Head.
- **Inputs**: 
  - `x`: Node features `(N, 4/5)`
  - `edge_index`: Graph connectivity `(2, E)`
  - `edge_attr`: (Optional) Edge features (length, capacity, speed) `(E, 3)`
  - `batch`: Graph batch indicator
- **Outputs**:
  - `probs`: 다음 이동할 이웃 노드에 대한 Softmax 기반 확률 분포.
  - `value`: 현재 상태의 Critic Value (V).
- **Features**: `use_jk_net` (Jumping Knowledge), `use_global_pool` (Global Mean Pooling) 등의 확장을 지원하여 수용장(Receptive Field) 확장을 제어.

#### `manager.py` (`Manager`)
- **Architecture**: GATv2(3 Layers) + Transformer Encoder(3 Layers) + Dual-Head Actor (Target/Zone) + Critic Head.
- **Inputs**:
  - `zone_features`: `[K, 6]` (GATv2 인코딩 입력)
  - `zone_edge_index`: `[2, E_zone]` (Zone 간 연결)
  - `target_features`: `[N, 6]` **(Phase 2A: 4→6)** — deadline, time_remaining, dist, urgency_ratio, feasibility, slack
  - `zone_dist_matrix`: `[B, K, K]` (최단 경로 물리적 거리, Dijkstra)
- **Context Generator (Phase 2B)**: `generate_context(elapsed, rescued, num_feasible, avg_urgency)` → `[B, 4]` → `context_proj(Linear(4, 256))` → query `[B, 256]`. **h_last 제거** (항상 0으로 노이즈만 추가하던 문제 해결).
- **Outputs**:
  - `target_logits`: 다음으로 향할 구출 대상 타겟 선택.
  - `zone_logits`: 선택된 타겟을 향해 나아갈 다음 Subgoal Zone.
- **Key Feature**: `zone_score_net`의 Logit 계산 시 `zone_dist_matrix` 거리를 직접 Concat하여 지리적 감각(Spatial Awareness) 확보.

### 1.3. Trainers (`src/trainers/`)
- **Trainer 기반 구조**: PPO (Proximal Policy Optimization)
- **`worker_trainer.py`**: Worker PPO 학습. GAE 기반 어드밴티지 계산. Normal (10%), Static (30%), Dynamic (60%) 단계의 재난 강화형 커리큘럼 적용.
- **`manager_trainer.py` (Phase 2C, 2F)**:
  - **Context 생성**: `generate_context(elapsed, rescued, num_feasible, avg_urgency)` — h_last 제거
  - **보상 체계 (Phase 2C)**:
    - 구출 보상: ×20.0 (기존 ×10.0)
    - PBRS: `(log1p(prev) - log1p(curr)) × 2.0` (기존 ×5.0, 스파스 보상 압도 방지)
    - 데드라인 만료 페널티: -5.0 per expired target (NEW)
    - 여유 시간 보너스: `min(slack/max_time × 5.0, 3.0)` (NEW)
  - **커리큐럼 (Phase 2E)**: 5 Phases, P1:Single → P2:Multi → P3:Static → P4:Dynamic → P5:Full
  - Entropy Bonus를 통해 탐색 장려. Learning Rate Scheduling 내장.

## 2. API Flow & Execution Pipeline

전체 시스템의 파이프라인은 크게 두 갈래의 Entry Point를 지닙니다.

1. **Zone 파티셔닝 (`generate_zones.py`)**:
   - `python src/utils/generate_zones.py --map Anaheim`
   - `data/Anaheim_node.tntp` 로드 $\rightarrow$ METIS 분할 $\rightarrow$ `grid_Anaheim_node_to_zone.json` 등 메타데이터 파일 배출.

2. **학습 루프 (`train_rl.py`)**:
   - **Stage: Worker**: `python scripts/train_rl.py --stage worker --map Anaheim`
   - **Stage: Manager**: `python scripts/train_rl.py --stage manager --map Anaheim`

3. **벤치마크 평가 (`evaluate_algorithms.py`)**:
   - `python scripts/evaluate_algorithms.py --episodes 100 --map Anaheim`
   - HRL(Neural), GA-Neural, ALNS-Neural, GA-Dijkstra, ALNS-Dijkstra 5개 모델에 대해 Rescue Rate, Latency, Recomputes, UGV Destroys 지표를 정량 비교. `src/models/heuristics.py`에 구현된 고전적 휴리스틱 매니저(GA, ALNS) 및 워커(Dijkstra)와 HRL 신경망을 동일한 시드 하에서 대결시켜 성능을 검증합니다.

4. **Worker 전용 벤치마크 (`compare_workers.py`)**:
   - `python scripts/compare_workers.py`
   - ALNS Manager 하에서 Worker 모델(Layer-2, Layer-3, Layer-4) 간의 자율 주행 성능(Recomputes 감소, Latency 단축 등)을 정량 평가.

## 3. Recent Architectural Refinements (2026-06-26)
*   **Manager Tensor State Sync (`hrl_env.py`)**: 재난 발생(`reset()`) 직후 `zone_dist_matrix`가 갱신되지 않던 문제를 해결하여 Manager가 실시간 피해가 반영된 물리적 최단거리를 참조하도록 아키텍처 수정.
*   **Per-Batch Aftershock Cursor (`hrl_env.py`)**: `aftershock_cursor`를 스칼라에서 `[B]` 형태의 독립 텐서로 변경하여 미니배치 내 개별 환경들의 동적 재난 스케줄 완전 격리.
*   **Worker Edge-Attribute Sync (`worker_trainer.py`)**: WorkerTrainer 미니배치 수집 시 `env._build_graph_data()`를 명시적으로 호출하여 GNN이 최신 데미지(Damage) 채널 값을 기반으로 Attention 가중치를 연산하도록 데이터 파이프라인 동기화 보장.
# Update: Node-Level Disasters, Dynamic Time-Windows, and Manager Urgency Reward

## Worker State (7-dim)
- **Shape**: `[B, N, 7]`
- **Channels**:
  - `0`: `is_curr` (1.0 if current node, 0.0 otherwise)
  - `1`: `is_tgt` (1.0 if node is target, 0.0 otherwise)
  - `2`: `zone_info` (1.0 if next zone, -1.0 if target zone, 0.0 otherwise)
  - `3`: `dist_to_tgt` (Normalized distance)
  - `4`: `dist_to_next_z` (Normalized distance)
  - `5`: `is_visited` (1.0 if visited, 0.0 otherwise - Anti-Loop)
  - `6`: `node_damage` (0.0 to 1.0, representing physical infrastructure collapse - Preemptive Detour)
  
## Manager Zone Features (7-dim)
- **Shape**: `[B, K, 7]`
- **Channels**:
  - `0`: `is_current`
  - `1`: `has_target`
  - `2`: `is_visited`
  - `3`: `disaster_intensity` (Edge-based Zone Average Damage)
  - `4`: `dist_from_curr`
  - `5`: `has_failed_target`
  - `6`: `zone_max_node_damage` (Node-based Zone Max Damage - 최댓값을 추출하여 지뢰밭 회피 유도)
  
## Dynamic Time-Window
When an aftershock occurs (`apply_aftershock` returns `affected_nodes`), the deadline of any target residing in the `affected_zones` shrinks by 20% to 40% uniformly:
`reduction = deadlines * random.uniform(0.2, 0.4)`
`deadline = max(current_time + 10.0, deadline - reduction)`

## Slack Bonus & Global Rescue Rate (Manager Reward)
Instead of rewarding the Manager for rescuing targets late (which caused reward hacking), we now use:
- **Slack Bonus**: `slack_bonus = 20.0 * (rem_time / tot_time)`. Rescuing a target earlier yields a higher bonus (max +20.0).
- **Global Rescue Rate Bonus**: `global_bonus = 50.0 * rescue_rate`. Granted at the end of the episode to emphasize overall success rather than local greedy rescues.
- **Base Rescue Reward**: Reduced to `10.0` to prevent inflation.

## Target KIA (Mission Evaporation) & Agent KIA
When an aftershock occurs, the `affected_nodes` dictionary returns the **Delta (instant increase in damage)**.
- **Agent KIA**: If the UGV's current node delta is $\ge 0.3$, the UGV is instantly destroyed (`agent_destroyed`).
- **Target KIA**: If a unrescued target's node delta is $\ge 0.5$, the building collapses, the target dies (`target_failed = True`), and the manager receives a `-5.0` penalty.
