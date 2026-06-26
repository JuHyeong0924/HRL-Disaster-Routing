# `src/models/heuristics.py` 로그

## 상세 코드 설명
- **역할**: 기존 HRL(신경망) 모델과 벤치마크 평가를 비교하기 위한 베이스라인(휴리스틱) 매니저/워커 알고리즘들을 제공합니다.
- **`GA_Manager`**: 
  - 유전 알고리즘(Genetic Algorithm)을 사용하여 타겟 방문 순서(TSP와 유사)를 최적화합니다.
  - 현재 시각(`current_time`)과 마감 기한(`deadlines`)을 평가 함수의 적합도(Fitness)에 반영하여 시간 제한 내 타겟 도달을 목표로 합니다.
  - 선택(Selection), 교차(Crossover), 변이(Mutation) 과정을 거쳐 최적의 염색체(`best_chromosome`)를 찾고, 첫 번째 타겟 노드가 속한 Zone을 다음 목표(`z_act`)로 지시합니다.
- **`ALNS_Manager`**: 
  - 적응형 대이웃 탐색(Adaptive Large Neighborhood Search)을 기반으로 휴리스틱 타겟 라우팅을 수행합니다.
  - 파괴 연산(Destroy)과 수리 연산(Repair)을 통해 해(Route)를 개선합니다. 
- **`Dijkstra_Worker`**: 
  - 매니저가 지시한 목표(Subgoal)를 향해 NetworkX의 `nx.shortest_path` (다익스트라 알고리즘)를 사용하여 한 홉(hop) 이동합니다.
  - 경로가 단절(`NetworkXNoPath`)되어 있다면 이동하지 않고 제자리(`c_idx`)를 반환합니다.
- **텐서 논리**: 
  - HRL `ManagerUnified`와 달리 딥러닝 텐서(Batch 등)를 적극 사용하지 않고 `batch_size=1` 환경에 특화되어 `[0]` 인덱스 단위의 텐서 연산 및 NetworkX 그래프 연산을 주로 활용합니다. `get_action()` 및 `get_actions()` 등 HRL_Env.step_manager 와 호환되도록 인터페이스를 설계했습니다.

## 시행착오 (Trial & Error)
- **에러**: 처음에 `Dijkstra_Worker`가 `__call__`에서 `edge_attr`를 받지 않아 `evaluate_algorithms.py` 평가 시 TypeError 발생.
- **해결 (`validation` 단계)**: `step_manager`에서 `self.worker.is_heuristic` 플래그를 확인하여, 휴리스틱 워커일 경우 `get_actions(active)` 메서드를 호출하도록 `HRLEnv` 와 인터페이스를 유연하게 호환시킴으로써 충돌을 방지했습니다.
