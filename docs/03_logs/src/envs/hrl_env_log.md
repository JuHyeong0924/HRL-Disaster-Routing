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

## 6. Phase 2 Manager Retraining Compatibility (v7.3)
- `ManagerTrainer` 구조는 싱글 타겟 `ManagerEnv`를 기준으로 설계되었기에 내부적으로 `self.env.zone_dist_matrix`를 호출하여 Zone 단위의 거리 지표를 상태 표현에 추가합니다.
- `train_manager.py`를 다중 타겟 환경(`HRLEnv`)에 직결 시 해당 프로퍼티가 없어 크래시가 발생하는 것을 방지하기 위해, `HRLEnv.__init__`에서 NetworkX `ZG` 그래프의 다익스트라(APSP)를 수행하여 `zone_dist_matrix` 텐서를 사전 계산하여 캐싱하도록 동적 확장을 구현했습니다.

## 7. Dynamic Disaster NaN Bug Fix (Isolated Targets)
- **문제점**: 동적 재난 시나리오(`dynamic_disaster=True`)나 원본 맵 상의 고립된 서브그래프 문제로 인해, 시작 위치에서 도달할 수 없는 노드가 무작위 타겟으로 선정되는 치명적 버그가 존재했습니다. 이 경우 다익스트라(Dijkstra) 거리가 `inf`로 반환되어 `log1p(inf) - log1p(inf) = NaN` 형태의 연산이 모델 내부로 전파되고 가중치를 폭파시켰습니다.
- **해결 방안 1 (방어 로직)**: `step_manager()` 내부 PBRS 보상 연산과 `get_target_features()` 피처 추출 시 `dist_matrix`가 `inf`이면 `self.env.max_dist`로 치환하는 1차 방어막을 구축했습니다.
- **해결 방안 2 (근본 수정)**: `reset()` 단계에서 단순히 `random.choice(nodes)`를 하는 대신, 출발 노드(`s_idx`)로부터 다익스트라 거리가 유한한(`dist < inf`) 즉, 물리적으로 도달 가능한 노드 리스트(`reachable`)를 미리 구한 뒤 그 중에서만 타겟을 추출하도록 전면 수정했습니다. 출발지가 너무 외진 곳이라 도달 가능한 노드 수가 타겟 개수보다 적으면 즉시 출발지부터 다시 뽑도록 강제하여 `NaN`의 근본 원인을 제거했습니다.
- **예외 처리 보강**: 또한 `step_manager()` 중 동적 재난으로 인해 엣지가 삭제되어 워커가 이동하려는 엣지가 사라지는 상황에 대비하여, `G.has_edge(c_node, v_node)` 체크 로직을 보강하여 `KeyError` 발생을 막았습니다.
