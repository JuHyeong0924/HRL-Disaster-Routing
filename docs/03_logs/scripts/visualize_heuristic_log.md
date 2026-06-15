# `scripts/visualize_heuristic.py` Log

## 1. 코드 상세 설명
- **역할**: 시뮬레이터(Manager + Worker)의 평가 루프를 시각화 목적으로 구동하는 독립적 실행 스크립트.
- **핵심 로직 변경 (Continuous Time Update)**:
  - 기존의 단순 Hop 반복문(`for i in range(len(path))`) 구조를 전면 폐기하고, **`0.5` 틱(Tick) 단위의 글로벌 타임(`global_time`) 기반 이벤트 스케줄러**로 개편하였습니다.
  - 워커가 노드 간 간선(Edge)을 이동하기 시작하면, 간선의 가중치(`edge_weight`)만큼 `worker_busy_time`을 설정하여 "이동 중" 상태에 돌입하며, 해당 시간 동안 의사결정이 잠깁니다.
  - **다중 동적 재난 트리거 로직**: 재난의 발동 조건 역시 스텝 횟수가 아닌 `global_time == 30.0`, `60.0` 등으로 물리적 시간에 맞춰 정확하게 발동하도록 동기화하였습니다.
  - **휴리스틱 매니저 로직 개선**:
    - **타겟 제외 (Failure Check)**: `global_time`이 `TW`를 초과한 타겟(`hrl_env.target_failed == True`)은 구출 대상으로 선정하지 않고 스킵.
    - **가중치 기반 최적화**: Manager의 목표 타겟 선정(`best_dist` 갱신) 과정에서, **`path_weight` (파괴도가 반영된 가중치 거리)**와 **`urgency` (남은 데드라인 임박도)**의 합산 기반 점수(`score`) 체계로 평가를 수행합니다.

## 2. 알고리즘 & 텐서 로직
- **Time/Event Loop**: 
  - `global_time`은 매 루프마다 `+0.5`씩 증가.
  - `worker_busy_time` 역시 `0.5`씩 차감되며, 0 이하가 되면 목표(next_node) 도착으로 간주.
  - 렌더링 측면에서 `progress = 1.0 - (worker_busy_time / total_edge_time)`을 계산해 `visualizer`로 선형 보간 진행도를 전달.
- **휴리스틱 수식**: `score = path_weight + urgency * 0.5`

## 3. Trial & Error (Troubleshooting)
- **이슈 1**: 노드 도착 시(`worker_busy_time <= 0`) `c_node`가 `next_node`로 업데이트되기 전에 `path_idx >= len(path) - 1` 조건문이 실행되어, 새로운 타겟 경로를 탐색할 때 마지막 엣지를 무한 반복해서 왕복하는 논리적 버그(텔레포트 현상) 발견.
  - **해결**: 조건문 상단에 `if c_node != next_node: c_node = next_node`를 명시적으로 추가하여, 물리적 도착 상태와 논리적 도착 상태를 완전히 동기화시킴.
- **이슈 2**: 워커가 이동할 때 이전처럼 칸(Step) 단위로 이동하는 것이 아니라, 가중치가 50.0배 늘어난 간선에서는 50.0의 시간을 대기해야 하므로, 스텝 반복문(`for _ in range(num_steps)`) 대신 글로벌 시간 시계(`global_time`) 중심의 이벤트 루프(While)로 아키텍처를 대대적으로 개편.
