# HRLEnv - Detailed Architecture & Refactoring Log

## 1. Overview
`HRLEnv`는 `ManagerEnv`와 상속/유사 관계로 설계된 계층형 통합 환경(Hierarchical Environment)으로, 매니저가 Zone 이동 명령을 내리면 내부 워커(Worker) 루프가 백그라운드에서 실행되어 맵 위를 물리적으로 이동합니다. 

## 2. Inner-loop Vectorization Optimization (v7.2)
매니저 1턴당 50스텝 이하를 반복 수행하는 Worker 구동 루프 `for _ in range(50):` 내부에는 256 배치 데이터를 뽑아내는 막대한 통신 비용이 존재했습니다.
- 기존 구현: `st = torch.stack([st[b] for b in active])`
  리스트 컴프리헨션과 `torch.stack`을 매 스텝(최대 12,800회/에피소드) 수행하면서 CPU 1300% 점유율이라는 치명적인 오버헤드를 발생시켰습니다.
- **최적화 구현**: `st = self.env._get_state_batch()` 와 `st[active].to(device)` 기반의 **Zero-Copy PyTorch Indexing**을 통해 파이썬 오브젝트 생성 부하를 C++ 텐서 메모리 연속 복사로 완벽히 대체했습니다. 

## 3. Zone Masking Vectorization
- **기존 `get_zone_mask`**: 256번 반복하며 NetworkX 그래프 딕셔너리에 접근. 
- **변경 `get_zone_adj_mask`**: `self.env._zone_adj_matrix_tensor`의 캐싱된 행렬 슬라이싱으로 즉각 추출. GPU 가속의 이점을 활용하여 1ms 이내로 256배치 전체 마스크를 확보하도록 구조를 개선했습니다.

## 4. Trial & Error (Device Synchronization)
`hrl_env.py`의 `reset()`은 `worker_env` 내부 변수인 `self.env.curr_nodes`를 보존/복원하는데, 이 과정에서 `torch.Tensor.clone().to(device)`를 쓰면서 의도치 않게 GPU 텐서로 타입 캐스팅이 고정되는 현상이 발견되었습니다. 이로 인해 `worker_env`의 `_node_zone_tensor` (CPU 할당본)과 연산 시 `RuntimeError`가 발생했습니다. 이를 방어하기 위해 인덱싱 수행 전 반드시 `tensor.cpu()`를 체이닝하여 안전성을 극대화했습니다.

## 5. Phase 1 Bug Fix (Manager-Turn Scoped Memory)
- **문제점**: 워커에 재방문 패널티(`-5.0`)를 추가한 결과, 다중 타겟 환경(Phase 2)에서 **과거 타겟으로 가기 위해 밟았던 정상적인 경로**마저 영구적으로 금지되어 길을 잃는 부작용이 발견될 뻔했습니다.
- **해결 방안 (Setup Loop Reset)**: 매니저가 새로운 구역(Zone)과 최종 목적지(Target)를 워커에게 부여하는 `step_manager()` 루프(Setup Loop) 진입 시, **워커의 과거 방문 이력(`visited_nodes`)을 깨끗하게 0.0으로 초기화**하도록 강제했습니다.
- **예외 처리**: 단, 워커가 현재 서 있는 현재 위치(Current Node)는 `1.0`으로 다시 마스킹하여, 이동 시작 직후 곧바로 뒤로 돌아가는 제자리 루프를 방지했습니다. 이를 통해 워커는 오직 **'현재 매니저가 부여한 단일 과제'**에만 집중하는 이상적인 HRL 철학을 유지할 수 있게 되었습니다.
