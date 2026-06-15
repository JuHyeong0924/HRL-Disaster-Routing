# 📝 Manager POMO Trainer Log (`src/trainers/manager_pomo_trainer.py`)

## 1. Intent & Purpose (Detailed Functional Specs)
*   **Role:** Manager 네트워크를 POMO(Policy Optimization with Multiple Optima) 방식으로 강화학습시키는 트레이너 클래스입니다.
*   **Architecture & Workflow:**
    1.  **State Duplication:** 배치 사이즈(`B_true`)의 환경을 `K`개 복제하여 `B_env = B_true * K` 크기의 환경으로 병렬 롤아웃을 준비합니다.
    2.  **First Action Forcing:** POMO의 핵심인 '다양한 궤적 탐색'을 위해 첫 번째로 방문할 Target을 `(k % num_targets)` 모듈러 연산으로 강제 선택합니다.
    3.  **Sequential Decoding:** `while` 루프를 돌며 진행되지 않은(Active) 환경들에 대해서만 Manager 네트워크가 타겟과 Zone을 순차적으로 예측합니다.
    4.  **Reward Baseline:** 롤아웃 종료 후 `[B_true, K]` 형태로 보상을 재배열하고, 각 배치 내 K개의 결과 평균을 Baseline으로 삼아 Shared Baseline Advantage를 계산 및 최적화합니다.

## 2. Tensor & Data Flow
*   **Target Embeddings:** `[B_env, num_targets, 128]` 크기 텐서로 Zone 임베딩과 타겟 피처를 융합.
*   **Query Generation:** 이전 상태 `h_last`, 정규화된 `elapsed_time`, `num_rescued` 비율을 Concat하여 `[B_env, 128]` 형태의 컨텍스트 쿼리를 생성.
*   **Action Logits:** 생성된 쿼리와 각 임베딩 간의 BMM(Batch Matrix Multiplication) 연산을 수행. 비활성 환경(Active Slicing)은 배제하여 연산량을 최소화함.

## 3. Trial & Error (Debugging History)

### [2026-06-15] 🐛 디버깅: POMO의 첫 번째 Zone 액션 강제 할당 문제 수정
*   **발생 이슈 (Logical Bug):** 초기 구현에서 첫 번째 타겟을 강제 할당함과 동시에, Zone 액션마저 `curr_zone_action = curr_target_zones`로 강제로 설정해버림.
*   **원인 분석:** 타겟이 멀리 떨어진 Zone에 있을 경우, 인접한 Zone을 징검다리 삼아 단계적으로 주행해야 하는 HRL의 핵심 기능(Waypoint Navigation)이 무시되고 Worker가 먼 Zone으로 곧바로 직행하려 하는 오류가 발생함.
*   **해결 및 수정:** 첫 번째 스텝에서 Target은 강제로 선택(`t_log_prob=0`)하되, Zone 예측은 **정상적으로 Manager 네트워크를 통과시켜 인접 Zone 확률 분포(`z_log_prob`)를 계산하고 기록**하도록 구조를 수정함.
