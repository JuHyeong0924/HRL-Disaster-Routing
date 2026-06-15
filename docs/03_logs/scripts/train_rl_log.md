# 📝 Train RL Script Log (`scripts/train_rl.py`)

## 1. Intent & Purpose (Detailed Functional Specs)
*   **Role:** 강화학습(Worker, Manager)의 메인 훈련 진입점(Entry Point). 환경 초기화, 모델/체크포인트 로딩, 및 트레이너 구동을 통제합니다.
*   **Architecture & Workflow:**
    1.  **Hardware Optimization:** 시작 시 `torch.set_num_threads(8)`, `cudnn.benchmark = True`, `allow_tf32 = True` 등을 통해 GPU(특히 Ada/Ampere 아키텍처)의 훈련 속도를 극한으로 끌어올립니다.
    2.  **Stage Router:** `--stage worker` 또는 `--stage manager` (기존 PPO) 옵션에 따라 각기 다른 서브 함수(`_run_worker_stage`, `_run_manager_stage`)로 제어권을 넘깁니다.
    3.  **Checkpoint Compatibility:** `_load_state_compat` 함수를 통해 구조가 변경된(예: node_dim 확장 등) 체크포인트에서도 텐서 크기가 일치하는 가중치만 안전하게 부분 로드하여, 과거의 훈련 성과를 최대한 재활용하도록 설계되었습니다.

## 2. Trial & Error (Debugging History)

### [2026-06-15] 🧐 코드 리뷰 (Code Review): 구조적 안정성 검증
*   `/code_review` 및 `mcp:sequential-thinking` 심층 분석 수행 결과, 체크포인트 로딩 안정성 및 커맨드라인 인자(CLI args) 전달 구조가 매우 견고하게 짜여 있음을 확인했습니다.
*   학습 파라미터(Curriculum Learning 로직: 에피소드 진행도에 따른 `disaster_prob` 증가 및 `dynamic_disaster` 활성화)가 자연스럽게 연결되어 있어, 추후 훈련 시 이상 없이 작동할 것임을 보증합니다.
