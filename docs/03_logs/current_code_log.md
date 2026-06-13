# Tier 3: Trial & Error Log (current_code_log.md)

## 1. Overview
이 로그는 `HRL-Disaster-Routing` 시스템의 코어 로직을 구성하며 겪은 Trial & Error 및 리팩토링 이력을 기록합니다. 향후 유사한 버그나 아키텍처 변경 시 참조용으로 활용됩니다.

## 2. Refactoring Log (2026-06-13)

### 2.1. File Restructuring
- **이슈**: 스크립트 파일들이 한 폴더(`scripts/`)에 혼재되어 있어 목적 파악이 모호함.
- **조치 (Refactoring)**:
  - `scripts/generate_zones.py` $\rightarrow$ `src/utils/generate_zones.py`로 이동. (데이터 전처리용 유틸리티 성격 강화). 이동 후 sys.path를 루트 디렉토리를 가리키도록 `os.path.dirname` 단계를 추가 수정.
  - `scripts/evaluate.py`, `scripts/evaluate_crossmap.py` $\rightarrow$ `tests/evaluate.py`로 단일화 및 이동.
- **결과**:
  - `evaluate.py` 하나에 `--cross_map` 플래그 인자를 추가하여 Flat/HRL 평가와 Zero-shot Cross-map 평가 기능을 통합. 관리 포인트 축소.
  - `train_rl.py`는 메인 학습 라우터 스크립트 성격으로 `scripts/`에 유지.

### 2.2. Known Issues & Fixes in Evaluation
- **무한 루프(Ping-Pong) 방지 로직 (Worker)**:
  - 평가 과정 중 워커(Worker)가 두 노드 사이에서 핑퐁(Ping-Pong)하며 이동하는 무한루프 버그가 관찰됨. (Deterministic 액션 선택 시 발생).
  - **해결책**: `tests/evaluate.py` 내부 시뮬레이션 루프에서 `past_node in path_nodes[-5:]` 인 경우 해당 노드의 선택 확률(Logit/Prob)을 `0.001`로 대폭 감소시켜 우회로를 찾도록 휴리스틱 페널티 부여 로직이 존재함.
- **Manager-Worker Turn Control**:
  - Manager가 너무 많은 턴을 소모하거나 Worker가 Zone 내에서 길을 못 찾는 문제.
  - **해결책**: 맵 크기(`env.num_nodes / env.k_zones`)에 비례하여 동적으로 `worker_c_max`와 `manager_max_turns`를 할당하도록 개선. (평가 모듈에 동적 할당 로직 적용됨).

### 2.3. HRL Phase Training Issues
- **Critic Architecture Mismatch**:
  - Worker 구조 변경(`use_global_pool`, `use_is_visited` 등 추가) 시 이전 체크포인트와 키가 불일치하여 로드 실패 이슈.
  - **해결책**: `train_rl.py` 내에 `_load_state_compat` 함수를 도입하여 호환되는 Weights만 안전하게 로드(Soft-load)하고, 구조가 바뀐 Critic은 Random Initialize하도록 처리함.

## 3. Pending & Future Work
- Manager가 빈 Zone(Empty Zone)을 선택하는 예외 상황 발생 시의 강력한 에러 핸들링 (현재는 Warning만 발생).
- Worker PBRS Reward Tuning: Node-level 도달 실패 시 Zone-progress에 대한 Dense Reward 비율 세부 튜닝.
