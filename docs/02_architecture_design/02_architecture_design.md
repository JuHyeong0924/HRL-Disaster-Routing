# Tier 2: Architecture Design (Mid - Detailed Design)

## 1. System Schemas & Modules

시스템의 코어 모듈들은 `src/` 내에서 각각의 뚜렷한 책임을 가집니다. Python의 Type Hinting과 명확한 API Contract를 준수하여 설계되었습니다.

### 1.1. Environment Modules (`src/envs/`)

#### `disaster_map.py` & `disaster_env.py`
- **역할**: TNTP 기반의 도로망 데이터를 NetworkX 그래프로 로드하고 관리합니다. 그래프 구조에 Zone 메타데이터를 통합하여 물리적 라우팅 환경의 기반을 제공합니다.
- **주요 기능**: `apply_disaster_damage()`를 통해 노드/간선에 재난(지진 등) 피해를 확률적으로 부여하며, **HAZUS 기반 가중치 패널티(Closed 10배, Danger 5배, Caution 3배, Normal 1.5배)**를 동적으로 반영합니다. `is_reset` 모드가 아닐 경우 재난 피해도는 기존 피해와 물리적으로 누적(`min(1.0, current + new)`)됩니다.

#### `worker_env.py` (`WorkerEnv`)
- **역할**: Worker 단독 학습(Phase 1)을 위한 라우팅 환경. 하나의 Zone 내부 혹은 근접 인접 Zone으로 이동하는 세부 노드 스텝을 제어. 실제 물리적 거리(Dijkstra APSP, `weight='weight'`)를 기반으로 한 상태 공간과 보상 체계를 제공.
- **State Schema (Tensor Shape: `[N, 4]` or `[N, 5]`)**: 
  - `is_curr` (Channel 0): 현재 노드 위치 (One-hot, 1.0 or 0.0)
  - `is_tgt` (Channel 1): 최종 목표 노드 위치 (One-hot, 1.0 or 0.0)
  - `zone_info` (Channel 2): Zone 정보를 스칼라 값으로 압축 (`-1`: 금지/이동 외 구역, `0`: 현재 소속 구역, `1`: 다음 이동해야 할 목표 Subgoal 구역)
  - `dist` (Channel 3): 목표까지의 물리적 최단 거리. 학습 안정성을 위해 정규화 (`torch.clamp(dists, max=100000.0) / max(self.max_dist, 1.0)` 또는 `use_relative_hop` 사용 시 상대적 거리 감소폭)
  - `is_visited` (Channel 4, 선택적): 과거 방문한 노드 여부(순환 궤적 방지용, `use_is_visited=True` 일 때 활성화)
- **Action Schema**: 현재 노드와 연결된 직접적인 이웃 노드(Neighbor Node ID). Masking을 통해 연결되지 않은 노드는 Logit = `-inf` 처리.
- **Reward Schema**: 
  - Sparse Reward (Goal): 최종 타겟 또는 목표 구역 도달 시 강력한 양의 보상(`+50.0`, `+5.0`) 부여.
  - Base Penalty (Time): 소모되는 물리적 거리(Edge weight)에 비례하는 지속적인 타임 패널티(`-0.1 * weight`).
  - Constraint Penalty (Revisit): 탐색 과정의 무한 루프(Stagnation)를 방지하기 위해, 이미 방문했던 노드(`visited_nodes == 1.0`)로 되돌아가려 할 경우 강력한 마이너스 보상(`-5.0`)을 즉각 부여.
  - Dense Reward (PBRS): 목표 노드까지의 물리적 거리를 기반으로 한 Potential-Based Reward Shaping. 거리가 줄어들면 양(+)의 보상, 멀어지면 음(-)의 보상을 주어 탐색 유도 (식: $\Phi(s) = max\_dist - dist\_to\_goal$).
- **Vectorization (병목 혁신)**: 기존 파이썬 기반 `for` 루프와 `torch.stack` 슬라이싱을 완전 폐기하고, 사전 캐싱된 `[N, N]` Dense Boolean Adjacency 텐서와 `batch_idx = torch.arange(B)`를 활용한 Zero-Copy PyTorch 고급 인덱싱 문법을 도입하여 C++ 백엔드 레벨에서 모든 환경의 상태, 보상, 마스크, 전이를 `O(1)` 병렬 연산으로 풀 벡터화(Full-Vectorization)하였습니다. 이로 인해 CPU Starvation이 사라지고 GPU UTL이 94%에 육박하는 극단적인 연산 최적화를 달성했습니다.

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
- **역할**: Manager가 여러 개의 타겟(Multi-OD)을 동시다발적으로 처리하고 데드라인(Deadline)을 관리할 수 있도록 래핑한 최상위 통합 환경. Worker와 Manager 모델을 동시에 탑재하여 전체 End-to-End 시뮬레이션을 통제합니다.
- **Key Breakthrough (Manager-Turn Scoped Memory)**: 워커에 적용된 '재방문 패널티(-5.0)'가 다중 타겟 환경에서 **과거 타겟을 찾기 위해 지나왔던 정상적인 교차로**마저 영구 통제구역으로 만들어버리는 버그를 막기 위해, 매니저가 새로운 목표를 하달하는 턴(Turn) 단위마다 **워커의 내부 방문 이력(`visited_nodes`)을 깨끗하게 0.0으로 초기화**합니다. 이를 통해 워커는 오직 '매니저가 지금 내린 단일 임무'에만 집중하는 이상적인 HRL 철학을 수학적으로 유지합니다.

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

#### `manager_unified.py` (`ManagerUnified`)
- **Architecture**: Multi-Level Encoder 구조. Local Topology는 GATv2(2 Layers)로 처리하고, Global Topology는 Transformer Encoder로 장기 의존성을 파악합니다. Dual-Head Actor (Target/Zone) + Critic Head.
- **Inputs**:
  - `zone_features`: `[K, 6]` (GATv2 인코딩 입력)
  - `zone_edge_index`: `[2, E_zone]` (Zone 간 연결)
  - `target_features`: `[N, 4]`
  - `zone_dist_matrix`: `[B, K, K]` (최단 경로 물리적 거리, Dijkstra)
- **Outputs**:
  - `target_logits`: 다음으로 향할 구출 대상 타겟 선택.
  - `zone_logits`: 선택된 타겟을 향해 나아갈 다음 Subgoal Zone의 Softmax 확률.
- **Key Breakthrough**: 기존의 단순 GNN 구조에서 벗어나, `zone_score_net`의 최종 Logit 계산 시 `zone_dist_matrix`의 거리(`target_dists` 차원: `[B, K, 1]`)를 직접 Concat하여 결합함으로써 수학적으로 확실하게 타겟 방향의 최단 경로 Zone을 선택할 수 있는 지리적 감각(Spatial Awareness)을 확보했습니다.

### 1.3. Trainers (`src/trainers/`)
- **Trainer 기반 구조**: PPO (Proximal Policy Optimization)
- **`worker_trainer.py` & `manager_trainer.py`**:
  - `rollout()` 단계에서 병렬 에피소드를 수집(Mini-batching).
  - GAE(Generalized Advantage Estimation)를 계산하여 Value 손실과 Policy 손실(Surrogate Objective)을 최적화.
  - Entropy Bonus를 통해 탐색(Exploration) 장려 (`entropy_coeff`).
  - Learning Rate Scheduling 기능 내장.

## 2. API Flow & Execution Pipeline

전체 시스템의 파이프라인은 크게 두 갈래의 Entry Point를 지닙니다.

1. **Zone 파티셔닝 (`generate_zones.py`)**:
   - `python src/utils/generate_zones.py --map Anaheim`
   - `data/Anaheim_node.tntp` 로드 $\rightarrow$ METIS 분할 $\rightarrow$ `grid_Anaheim_node_to_zone.json` 등 메타데이터 파일 배출.

2. **학습 루프 (`train_rl.py`)**:
   - **Stage: Worker**: `python scripts/train_rl.py --stage worker --map Anaheim`
     $\rightarrow$ `WorkerEnv` 초기화 $\rightarrow$ `Worker` 모델 무작위 가중치 생성 $\rightarrow$ `HRLWorkerTrainer` 호출 $\rightarrow$ `logs/rl_worker_stage/` 에 `best.pt` 체크포인트 기록.
   - **Stage: Manager**: `python scripts/train_rl.py --stage manager --map Anaheim`
     $\rightarrow$ 이전 단계의 `best.pt` Worker 로드 및 동결 $\rightarrow$ `ManagerEnv` (Closed-loop) 초기화 $\rightarrow$ `ManagerTrainer` 호출 $\rightarrow$ `logs/rl_manager_stage/` 에 Manager `best.pt` 기록.

3. **평가 루프 (`evaluate.py`)**:
   - `python tests/evaluate.py --map all --cross_map`
   - 학습된 체크포인트를 불러와 HRL 환경에서 성능 평가. Cross-map 모드시 재학습 없이 Manager/Worker 모델이 다른 맵(Zero-shot)에서 라우팅 성능(Success Rate, Path Expansion Ratio 등)을 도출 및 시각화(`hrl_path_*.png` 출력).
