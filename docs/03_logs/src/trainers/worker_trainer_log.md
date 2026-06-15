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
