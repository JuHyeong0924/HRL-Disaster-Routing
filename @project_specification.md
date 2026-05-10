# Project Specification: HRL Disaster Routing

## 1. HRL Worker Architecture (`src/models/worker.py`)

### 1.1. State & Input Features
- **Node Dimension (`node_dim`)**: 기본 4-Dim 
  - `[is_curr, is_tgt, is_next_zone, hop_dist]`
- **Edge Dimension (`edge_dim`)**: 기본 3-Dim (Edge-Conditioning 활성화 시)
  - `[length, capacity, speed]` (Min-Max 정규화 적용됨)

### 1.2. Spatial Encoder (GNN)
- **Base Model**: `GATv2Conv` (PyG) + `GraphNorm` + Residual Connection
- **Layers**: CLI 인자 `--num_layers`로 동적 할당 (기본 2~3)
- **Feature Shape**:
  - Input: `[Batch*N, node_dim]`
  - Hidden: `[Batch*N, hidden_dim]` (기본 256)
- **JK-Net (Jumping Knowledge)**:
  - `--use_jk_net` 활성화 시, 각 레이어의 출력값을 리스트에 모은 후 병합
  - 병합 후 `self.jk_proj`를 거쳐 `hidden_dim`으로 축소
  - 연산: `h_jk = Linear( torch.cat([h0, h1, ...], dim=-1) )`
- **Edge-Conditioned MP**:
  - `--use_edge_attr` 활성화 시, Message Passing 과정에 `edge_attr` 주입
  - 이를 통해 GATv2의 Attention Score(`\alpha_{ij}`) 계산 시 위상 정보 외 물리적 비용(비율/거리) 반영.

### 1.3. Temporal & Policy/Critic Heads
- **Temporal Proj**: LSTM을 배제하고 단순 `Linear + ReLU`로 치환. VRAM 절약 및 메모리 병목 회피.
- **Scorer (Policy)**: `[N, hidden_dim * 2]` 입력 (현재 노드 임베딩 + 타겟 컨텍스트) -> `[N, 1]` Logit 출력. Softmax 및 Action Masking 거침.
- **Critic (Value)**: `[1, hidden_dim]` 입력 -> `[1, 1]` State Value 출력.

---

## 2. Worker Trainer (`src/trainers/worker_trainer.py`)

### 2.1. Environment Data Loading
- **Edge Features**: `DisasterMap` 객체(`env.dm`)에서 정적 엣지 속성(`capacity`, `length`, `speed`)을 가져와 리스트 구축 후 양방향(Bidirectional) 할당.
- **Normalization**: 추출된 피처들은 `(x - min) / (max - min)`으로 [0, 1] 범위로 스케일링.

### 2.2. Training Loop (`_run_batch_episodes`)
- **Parallelism**: POMO 기반의 `batch_size`를 `env.reset(batch_size=B)`로 적용하여 동시 전개.
- **Gradient Accumulation**: `batch_size` 묶음을 한 번에 GNN Forward 처리하여 분산을 낮추고 학습 안정성을 높임.
- **Loss Computation**: 
  - GAE(λ) 기반 Advantage 계산 옵션 지원 (`--use_gae`). 미사용 시 단순 MC Return 적용.
  - Policy Loss, Value Loss, Entropy Bonus 통합.
- **Data Flow Shape**:
  - `x_flat`: `[|Active_Batch| * N, 4]`
  - `batch_ei`: `[2, |Active_Batch| * E]` (Graph 별 Offset 추가됨)
  - `batch_ea`: `[|Active_Batch| * E, 3]` (`edge_attr.repeat` 처리)

---

## 3. Subgoal Mode (Planned)
- 향후 Zone 단위와 Node 단위의 Subgoal 방식을 스위칭할 수 있도록 `HRLZoneEnv`에 `subgoal_mode` (Zone / Node) 플래그 추가 예정.
- PBRS Reward 대상이 "최종 목적지"에서 "현재 할당된 Subgoal"로 전환되는 로직이 구현될 예정.

---

## 4. Hardware Optimization
- **CPU Threads**: 스레드 병목 및 컨텍스트 스위칭 오버헤드 방지를 위해 파이썬 프로세스당 PyTorch CPU 스레드 사용량을 8개로 하드코딩 적용 (`torch.set_num_threads(8)`, `OMP_NUM_THREADS="8"`). 병렬 학습 시 총 16코어 수준을 안정적으로 점유.
