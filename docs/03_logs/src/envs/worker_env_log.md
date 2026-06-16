# WorkerEnv - Detailed Architecture & Refactoring Log

## 1. Overview
`WorkerEnv`는 계층적 라우팅 환경의 하위 계층(Worker)의 학습 환경을 담당합니다. 초창기에는 파이썬 리스트 컴프리헨션과 `torch.stack()`을 활용하여 각 배치(Agent) 별로 순차적(Sequential)인 상태 변환 연산을 수행했습니다. 하지만 Manager와 함께 동작하는 `HRLEnv` 구조 안에서 초당 수만 개의 그래프 어텐션 스텝(GATv2)이 처리되어야 할 때, 이러한 파이썬 루프는 극단적인 **CPU Starvation (1300% 로드율)**과 **GPU Util 저하 (< 20%)**라는 성능 병목을 발생시켰습니다.

## 2. Vectorization Optimization (v7.2)
이 병목을 부수기 위해 `_get_state_batch` 및 `get_action_mask_batch` 함수를 C++ 백엔드에서 일괄 처리되는 **풀 벡터화(Full Vectorization)** 아키텍처로 개편했습니다.

### 2.1. 사전 캐싱 (Initialization)
- `self._adj_matrix_tensor`: `[N, N]` 크기의 Boolean Dense Tensor를 GPU에 할당하여, 노드 간 물리적 연결성을 `O(1)` 슬라이싱으로 즉시 파악합니다.
- `self._zone_adj_matrix_tensor`: Manager Action Masking 전용으로 `[K, K]` 인접 행렬 텐서를 GPU에 상주시켜, CPU의 NetworkX 탐색 부하를 원천 제거했습니다.
- `self._node_zone_tensor`: `[N]` 크기 텐서를 `device` 파라문을 통해 GPU에 바로 탑재함으로써, 상태 벡터 추출 시 CPU-GPU간의 통신 페널티를 없앴습니다.

### 2.2. O(1) Tensor Operations
- 기존의 `for b in range(B): state[b, curr_nodes[b]] = 1.0` 코드는 파이토치의 Advanced Indexing `state[batch_idx, curr_nodes, 0] = 1.0` 단 한 줄로 교체되었습니다.
- PBRS 보상을 위한 Dijkstra `dist_matrix` 조회 역시 `dist_tensor[curr_nodes, target_nodes]` 와 같은 인덱스 배열 전송 기법을 사용하여, 256명 분량의 로그 스케일 거리 정규화를 GPU 커널 위에서 단숨에(1ms 이하) 처리해 냅니다.

## 3. Trial & Error (Debugging Memory)
- **Device Mismatch Error**: 벡터화 도중 `hrl_env.py`의 `reset()` 로직 내부에서 `.clone()` 을 수행하며 텐서가 GPU로 자동 캐스팅되어, CPU 상에 남아있던 배열과의 형 불일치 인덱스 에러(`RuntimeError: indices should be either on cpu or on the same device`)가 발생했습니다.
- **해결 방안**: `get_zone_adj_mask` 메서드 내에서 인덱스 텐서를 `.cpu()`로 명시적 캐스팅함으로써 불확실한 Device 캐스팅 버그를 완전히 차단했습니다. 최종적으로 94%의 극한 GPU 활용률(Utilization) 달성에 성공했습니다.

## 4. Phase 1 Worker Retraining (Revisit Penalty)
- **무한 루프 버그 발생**: 이전 버전에서는 워커가 벽(끊어진 다리 등)에 막혔을 때, 이미 방문했던 노드를 반복해서 맴도는(Stagnation) 심각한 버그가 있었습니다.
- **해결 (Revisit Penalty 추가)**: `step_batch` 내부에서 `visited_nodes == 1.0`인 노드를 향해 이동하려고 할 경우 즉각적으로 **-5.0**의 거대한 패널티를 부여하도록 수학적 제약을 걸었습니다.
- **결과 (PPO의 진화)**: 워커가 초기에는 엄청난 패널티 폭탄(-47점)을 맞았으나, 10,000 에피소드 이후 "한 번 밟은 땅은 다시 밟지 않는다"는 룰을 스스로 깨닫고 일직선 주행(Rw=52.9)을 마스터하며 무한 루프가 근절되었습니다.

## 5. Unreachable Target Bug Fix
- **문제점**: `reset()` 시 출발지와 목적지(Target)를 순수하게 무작위로 추출(`random.choice`)하다 보니, 맵 내에서 서로 단절된(Disconnected) 서브그래프 간의 노드가 배정되는 문제가 있었습니다. 이는 모델에게 아예 클리어가 불가능한 목표를 부여하는 셈이었습니다.
- **해결 방안**: 시종착점을 선정할 때 단순히 `s != t`만 비교하는 것이 아니라, 다익스트라 거리 기반의 `self.dist_matrix[s_idx, t_idx] < float('inf')` 조건 검사를 추가하여 물리적으로 반드시 도달 가능한 목적지만 생성하도록 구조적 결함을 보완했습니다.
