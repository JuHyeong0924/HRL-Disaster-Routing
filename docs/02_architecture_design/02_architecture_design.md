# Tier 2: Architecture Design (Mid - Detailed Design)

## 1. System Schemas & Modules

시스템의 코어 모듈들은 `src/` 내에서 각각의 뚜렷한 책임을 가집니다. Python의 Type Hinting과 명확한 API Contract를 준수하여 설계되었습니다.

### 1.1. Environment Modules (`src/envs/`)

#### `disaster_map.py` & `disaster_env.py`
- **역할**: TNTP 기반의 도로망 데이터를 NetworkX 그래프로 로드하고 관리합니다. 그래프 구조에 Zone 메타데이터를 통합하여 물리적 라우팅 환경의 기반을 제공합니다.

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
  - Sparse Reward: 최종 도달 시 보상 부여.
  - Dense Reward (PBRS): 목표 노드까지의 물리적 거리를 기반으로 한 Potential-Based Reward Shaping. 거리가 줄어들면 양(+)의 보상, 멀어지면 음(-)의 보상을 주어 탐색 유도 (식: $\Phi(s) = max\_dist - dist\_to\_goal$).

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
  - PBRS: Zone 이동 시 목표와의 물리적 거리가 가까워진 정도에 비례하는 포텐셜 보상. 무한 핑퐁 루프 등 Negative Potential Exploit를 막기 위해 포텐셜이 음수로 발산하지 않도록 `min(dist, 50000.0)` 처리 등 수학적 안정장치가 적용됨.

### 1.2. Neural Network Models (`src/models/`)

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
- **Architecture**: Zone 레벨의 매크로 그래프(Zone Graph)를 처리하는 GATv2 모델 + Actor-Critic Head.
- **Inputs**:
  - `x`: Zone features `(K, 7)`
  - `edge_index`: Zone connectivity `(2, E_zone)`
- **Outputs**:
  - `action_probs`: 다음 이동할 Subgoal Zone에 대한 Softmax 확률 분포.
  - `state_value`: 현재 Zone State의 Critic Value.

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
