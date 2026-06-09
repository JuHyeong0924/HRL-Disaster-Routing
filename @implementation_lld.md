# HRL Implementation (Low-Level Design)

본 문서는 코드 스페이스의 모든 주요 클래스와 메서드의 서명(Type Hints), 상태 텐서(State Tensor)의 변화 추이(Shape Tracking), 그리고 핵심 수식 논리를 상세히 기록한 마이크로 레벨의 개발자 참조서(LLD)입니다.

---

## 1. Grid Partitioner (`grid_partitioner.py`)
### 알고리즘 요약
1. $V$ (노드 수) 기반 적정 격자 차원 계산: $N = \lceil \sqrt{V / 16} \rceil$
2. 맵 경계 좌표(BBox) 획득 및 격자 셀 크기 $dx, dy$ 계산.
3. 노드 위치(x, y)를 $(gx, gy)$ 인덱스로 매핑하여 `node_to_zone_grid.json` 생성.
4. 원본 Node Edge(도로) 중, 서로 다른 Zone에 속한 노드를 연결하는 엣지들만 추출하여 `zone_graph_grid.json` (Zone 간 연결성) 구축.

---

## 2. Worker Domain (`worker.py` & `worker_env.py`)

### 2.1. Tensor Shapes in Worker Model
입력 텐서 `x`의 초기 형태는 `[N, 5]` 입니다. GATv2Conv를 거치면서 Node Embedding으로 변환됩니다.
*   **Input Features (`[N, 5]`)**
    *   `x[:, 0]`: `is_curr` (현재 노드면 1.0, 아니면 0.0)
    *   `x[:, 1]`: `is_tgt_node` (최종 목표 노드면 1.0)
    *   `x[:, 2]`: `is_next_zone` (매니저가 지시한 Subgoal Zone 소속이면 1.0)
    *   `x[:, 3]`: `node_hop_dist` (현재 목적지까지의 A* 기준 홉 거리, Max Hop으로 정규화됨)
    *   `x[:, 4]`: `is_visited` (현재 에피소드 중 이미 방문했던 노드면 1.0)
*   **Edge Index**: `[2, E]`
*   **Global Pooling Output**: `global_max_pool(h, batch)` $\rightarrow$ `[Batch, Hidden]`
*   **Final Output**: `Actor Logits [N, 1]` 및 `Critic Value [Batch, 1]`

### 2.2. Method Signatures
```python
class WorkerEnv(gym.Env):
    def __init__(self, G: nx.Graph, config: dict):
        self.masking_mode = 'soft_curr_next' # 물리적 장벽은 없애고, 구역 이탈 시 OOB 페널티 부여
        
    def get_obs(self) -> torch.Tensor:
        # returns Tensor of shape [N, 5]
        
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        # Reward Logic (PBRS):
        # 1. Base step penalty = -0.1
        # 2. Potential diff = phi(curr) - phi(next)
        # 3. OOB Penalty = -1.0 (If next node NOT in Subgoal Zone & Not current zone)
        # 4. Target Reach Reward = +50.0
```

---

## 3. Manager Domain (`manager.py` & `manager_env.py`)

### 3.1. Tensor Shapes in Manager Model
입력 텐서 `x`의 초기 형태는 `[K, 4]` 입니다. ($K$ = 전체 Zone의 개수)
*   **Input Features (`[K, 4]`)**
    *   `x[:, 0]`: `is_curr_zone` (워커가 현재 속한 구역이면 1.0)
    *   `x[:, 1]`: `is_goal_zone` (최종 목적지가 있는 구역이면 1.0)
    *   `x[:, 2]`: `is_visited_zone` (이전 스텝에서 이미 거쳤던 구역이면 1.0)
    *   `x[:, 3]`: `zone_hop_dist` (Zone 레벨의 정규화된 홉 거리)
*   **Zone Edge Index**: `[2, E_zone]` (물리적 연결이 있는 구역 간의 간선)
*   **Action Candidate Mask**: `[K]` (현재 구역과 엣지로 연결된 인접 구역들에 대해서만 1.0 부여. 불법 이동 원천 차단)

### 3.2. Method Signatures
```python
class ManagerEnv(gym.Env):
    def get_candidate_mask(self) -> torch.Tensor:
        # returns Tensor [K], Valid adjacent zones = 1.0, others = 0.0
        
    def select_action(self, x, edge_index, edge_attr, curr_idx, goal_idx, deterministic=False):
        # Evaluation 시 무한루프 방지를 위해 Worker는 deterministic=False 유지
        # ...
        
    def _execute_worker(self, subgoal_zone_idx: int) -> List[int]:
        # 워커를 깨워서 Node Graph 상에서 물리적으로 이동시킴.
        # 루프 한도: c_max (20 steps). 도달 혹은 초과 시 제어권 반환.
        
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        # action은 선택된 subgoal_zone_idx.
        # Worker 실행 후, Worker가 최종 안착한 노드의 Zone을 self.current_idx 로 즉시 동기화(Dynamic Re-planning의 핵심).
        # Reward Logic:
        # 1. PBRS = (phi_after - phi_before) - (step_count * 0.1)
        # 2. Final Goal Reach = +100.0
```

---

## 4. Training Engine & Checkpointing (`train_rl.py`)

### 4.1. Memory Management for Large Graphs
*   **`batch_size` (예: 32)**: `train_rl.py` 단에서 파서로 입력받는 인자. 병렬 에피소드 수(Parallel Envs)를 의미함. 데이터 수집(Rollout)의 규모를 결정.
*   **`mini_batch_size` (예: 128)**: 거대한 맵을 학습할 때, 수집된 수만 장의 궤적(Trajectories)을 한 번에 PPO 업데이트에 넣으면 어텐션 가중치 행렬 $E \times Heads$ 연산 중 VRAM이 폭발함(OOM). 이를 막기 위해 PPO 업데이트 함수 내부에서 `x_flat` 텐서를 잘게 쪼개어 그래디언트를 누적시키는 크기.

### 4.2. Configuration Mapping
```python
def _get_config(args, loaded_checkpoint_paths):
    return Config(
        lr=args.lr,
        num_pomo=args.batch_size,     # 수집용 병렬 환경 수
        episodes=args.episodes,
        mini_batch_size=args.mini_batch_size, # VRAM 방어용 미니배치
        save_dir=save_dir,
        ...
    )
```

### 4.3. Script Execution Flow
1. **Worker Stage (`--stage worker`)**: `WorkerTrainer` 인스턴스화. GAE 연산 및 `_update_ppo` 호출을 통해 하위 정책망 훈련.
2. **Manager Stage (`--stage manager`)**: `--worker_ckpt`로 앞서 학습된 워커의 가중치를 불러와 고정(Freeze)시킨 뒤, `ManagerTrainer`를 통해 상위 정책망 훈련.
3. 모든 모델의 가중치는 `best.pt`와 `latest.pt`로 `logs/rl_*(stage)/...` 디렉토리에 저장됨.
