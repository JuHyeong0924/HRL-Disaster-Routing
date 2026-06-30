# 📝 Compare Workers Log (`scripts/compare_workers.py`)

## 1. Intent & Purpose
*   **Role**: 여러 종류의 Worker 체크포인트(Layer 2, 3, 4 등) 간의 성능을 정량적으로 비교(Benchmarking)하는 독립 스크립트.
*   **Mechanism**: 완전 학습된 상위 Manager(RL) 대신, Heuristic Manager인 **ALNS_Manager**를 사용하여 오직 Worker의 순수 로컬 회피 및 주행 능력만을 평가함.

## 2. Trial & Error (Debugging History)
### [2026-06-26] 구현 및 버그 수정
*   **버그**: `WorkerModel` 클래스 이름 오타 및 `edge_dim` 인자 전달 오류. 체크포인트 딕셔너리 로딩 키 오류.
*   **수정**: 올바른 `Worker` 클래스 사용, `node_dim=6`, `hidden_dim=256` 인자 고정, `load_state_dict(checkpoint)` 직접 매핑.
*   **결과**: 20 에피소드 평균 Layer-4가 Layer-2 대비 Recomputes를 약 20% 감소시키고, 평균 도착 지연 시간을 대폭 개선함을 증명.
