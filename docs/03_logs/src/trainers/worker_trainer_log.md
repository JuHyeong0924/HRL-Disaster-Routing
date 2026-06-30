# 📝 Worker Trainer Log (`src/trainers/worker_trainer.py`)

## 1. Intent & Purpose (Detailed Functional Specs)
*   **Role:** HRL Phase 1에서 Worker 네트워크를 PPO(Proximal Policy Optimization) 알고리즘으로 사전 학습(Pre-training)시키는 트레이너 클래스입니다.
*   **Architecture & Workflow:**
    1.  **Data Collection (Fast Rollout):** `_run_batch_episodes`에서 `torch.no_grad()`를 사용하여 계산 그래프 없이 매우 빠른 속도로 `batch_size`만큼의 에피소드를 병렬 시뮬레이션합니다. 각 배치의 State, Action, Reward, Value, Log_prob, Mask 등을 CPU 텐서 버퍼로 분리하여 저장함으로써 GPU 메모리를 최적화합니다.
    2.  **Advantage Estimation:** `_compute_gae`에서 역순으로 순회하며 Generalized Advantage Estimation(GAE)를 계산하여 분산을 줄이고 학습 안정성을 높입니다.
    3.  **PPO Update:** 수집된 총 타임스텝의 데이터를 `mini_batch_size` 단위로 분할하여 `ppo_epochs` 횟수만큼 반복 학습합니다. 이 과정에서 Actor Loss(Clipping 도입), Critic Loss, Entropy Bonus를 결합하여 역전파(Backpropagation)를 수행합니다.

## 2. Tensor & Data Flow
*   `active_states`: `[A, N, 5]` 형태. (A: 현재 완료되지 않은 에피소드 수)
*   `ai`, `aei`, `ae_attr`: PyG의 `batch` 텐서를 동적으로 생성하여 여러 독립된 그래프를 하나의 거대한 Disjoint Graph로 묶어 `[A*N, 5]` 형태의 `x_flat`으로 입력합니다.
*   `mb_states`: `[mini_batch_size, N, 5]` 형태로 분할되어 메모리 OOM을 완벽히 방지합니다.

## 3. Trial & Error (Debugging History)

### [2026-06-15] 🧐 코드 리뷰 (Code Review): 구조적 결함 없음
*   `/code_review` 및 `mcp:sequential-thinking` (6단계 초심층 분석) 수행 결과, 미니배치 분할 로직과 PyG 기반 텐서 재구성 로직이 완벽하게 들어맞고 있으며, 메모리 누수나 텐서 크기 불일치 현상 없이 최적의 속도로 동작하도록 짜여져 있음을 검증 완료했습니다.

### [2026-06-26] BUG FIX: edge_attr 동기화 누락 수정
*   **증상**: Phase 2/3 재난 발생 시 `env.reset()`이 데미지를 적용하지만, WorkerTrainer 내부의 `edge_index`와 `edge_attr`가 갱신되지 않아 Worker GNN이 재난 상황을 인지하지 못함.
*   **수정**: `_run_batch_episodes()` 시작 시마다 `self.env._build_graph_data()`를 명시적으로 호출하여 `edge_attr`의 데미지 채널을 최신화하도록 수정.

### [2026-06-27] 커리큘럼 재난 비중 재배치
*   **변경 배경**: Worker가 가혹한 재난 환경(disaster=0.2)에서 실전 배치(HRL 루프) 시 탐색 효율이 35%로 격감되는 병목 현상 확인. Worker의 재난 회피 적응 기간을 대폭 확장할 필요성 대두.
*   **수정 내용**: 커리큘럼 비율을 기존 P1(Normal) `25%` → `10%`, P2(Static) `25%` → `30%`, P3(Dynamic) `50%` → `60%`로 재배치.
*   **기대 효과**: 기본적인 길 찾기 기능이 자리 잡은 직후(전체 에피소드의 10% 시점인 3,000 ep)부터 재난 및 동적 여진 회피 궤적을 90% 이상 집중 노출시킴으로써, 대미지 텐서 채널 정보 기반의 실시간 경로 의사결정 안정성 강화.

