# hrl_env.py — 변경 로그

## Phase 1B: 시간 기반 Continuous Aftershock (2026-06-24)

### 변경 개요
Manager 턴 기반 여진 스케줄 → `current_time` 축 기반 스케줄로 전환.

### 핵심 로직
- `reset()`: `aftershock_times = sorted([interval*(i+1) + jitter for i in range(8~15)])`, `aftershock_cursor = 0`
- `step_manager()` Worker 루프: `while current_time >= aftershock_times[cursor]` → `dm.apply_disaster_damage(0.03)` + `need_rebuild = True`
- 근거: Omori's Law — 여진은 시간 축에서 독립 발생

## Phase 1C: UGV 파괴 판정 (2026-06-24)

### 핵심 로직
- Worker step 직후 `prev_n → new_n` 간선의 `status` 확인
- `Closed` → `random() < 0.3` → `agent_destroyed`, `ugv_destroys[b] += 1`
- `Danger` → `random() < 0.1` → `agent_trapped`, `current_time += 30.0`, 복귀
- 기존 Zone 전환 시 Trap 판정 코드 제거 (Phase 1C로 통합)

## Phase 2A: Target Features 확장 (2026-06-24)

### 변경 개요
`get_target_features()`: `[B, N, 4]` → `[B, N, 6]`

### 채널 설명
| Ch | 이름 | 수식 | 의미 |
|----|------|------|------|
| 0 | deadline | `d / max_time` | 데드라인 정규화 |
| 1 | time_remaining | `(d - current_time) / max_time` | 잔여 시간 |
| 2 | dist_from_curr | `dist / max_dist` | 현재 위치~타겟 거리 |
| 3 | urgency_ratio | `min(dist/rem, 5.0)` | 긴급도 (높을수록 위험) |
| 4 | feasibility | `1.0 if rem>0 and dist<rem else 0.0` | 도달 가능 여부 |
| 5 | normalized_slack | `clamp((rem-dist)/max_time, -1, 1)` | 여유 시간 |

### Trial & Error
- **오류 없음**: 모든 구현이 첫 시도에 통과

## 최적화 단계: CPU/GPU 병목 제거 (2026-06-24)

### 변경 개요
*   `step_manager()` 내 `get_action_mask_batch` 다중 호출 중복을 제거하여 O(A * B * N) 시간 복잡도를 O(B * N)으로 완화.
*   `edge_index`와 노드 정보(`bidir_nodes`)를 `__init__`에서 사전 캐싱하여, `_build_graph_data` 호출 시 유발되던 텐서 재생성 및 딕셔너리 스캔 오버헤드를 완전 제거.

### Trial & Error
*   `batch_size=256` 설정 시 30분 동안 연산이 중단된 것처럼 보였으나, 비효율적 마스킹 반복 루프(1.3억 회 이상 수행)로 인한 CPU 병목이 원인이었음.
*   최적화 반영 후 `batch_size=32` 기준 첫 배치 수집 속도가 75초에서 **17초**로 단축(약 4.4배 가속).

## 최적화 2단계: 피처 추출 완전 텐서화 및 캐싱 (2026-06-25)

### 변경 개요
*   WorkerEnv에서 도입된 Matrix Tensor를 활용하여 `get_zone_features`, `get_target_features` 등에서 사용되던 3중 파이썬 루프를 행렬 곱셈(`@`) 및 Fancy Indexing 연산으로 완전 벡터화.
*   `step_manager()` 루프 내 NetworkX (`G.has_edge`, `G[u][v]`) 호출을 `_adj_matrix_tensor`, `_weight_matrix` 룩업으로 교체.
*   `_build_graph_data`에 사용되는 `edge_attr`를 Python List 순회를 거치지 않고 Tensor 차원 슬라이싱으로 즉시 생성할 수 있도록 `_bidir_src`, `_bidir_dst` 인덱스 텐서 사전 캐싱.

### Trial & Error
*   벡터화 이후 `verify_optimizations.py`를 통해 140번 이상의 HRL step(약 7000회 이상의 워커 내부 스텝)이 **약 7초(스텝당 0.05초 미만)** 내에 연산되는 미친 성능(C++ 네이티브 레벨)을 확보함. CPU 점유율 이슈 완전 해소.

## Bug Fixes: Manager Tensor State Sync (2026-06-26)

### 1. zone_dist_matrix 미갱신 버그 수정
- **증상**: 재난(Disaster) 발생 후 `HRLEnv.reset()`에서 Zone 가중치는 바뀌나 최단거리 행렬(`zone_dist_matrix`)이 갱신되지 않아 Manager가 잘못된 거리로 타겟을 선정함.
- **수정**: `reset()`의 마지막에 `self.zone_dist_matrix = self.get_zone_dist_matrix_tensor()`를 호출하여 재계산하도록 수정.

### 2. aftershock_cursor 배치 공유 버그 수정
- **증상**: `batch_size > 1`일 때 `aftershock_cursor`가 스칼라로 선언되어, 첫 번째 배치가 여진을 겪으면 나머지 배치는 여진을 건너뛰는 현상 발생.
- **수정**: `aftershock_cursor`를 `[B]` 형태의 텐서(`torch.zeros(B)`)로 변경하여 각 배치마다 독립적인 여진 스케줄을 추적하도록 수정.

## Phase 1D: 여진 빈도 및 강도 강화 (2026-06-27)

### 변경 개요
가혹한 재난 환경(disaster=0.2)에서 UGV 파괴 지표의 유의미한 변동성을 확인하고 RL 모델의 위험 회피 우위를 평가하기 위해 여진 스케줄 빈도와 대미지 발생 비율을 상향 조정함.

### 핵심 로직
- **여진 발생 횟수 증가**: `reset()` 시 생성되는 여진 횟수를 기존 `8~15회`에서 `15~25회` 범위로 상향 조정.
- **여진 강도 증가**: 여진 1회 발생 시 대미지 전파 확률인 `apply_disaster_damage` 파라미터 `damage_prob`을 `0.03`(3%)에서 `0.05`(5%)로 상향.

### Trial & Error
- **레거시 테스트 실패**: `aftershock_times` 도입 이후에도 기존의 턴 기반 속성인 `aftershock_schedule`을 참조하여 테스트를 진행하던 `tests/test_hrl_env_dynamic.py`에서 Assertion Error 발생.
- **해결 방안**: 테스트 코드를 수정하여 새로운 시간 기반 API인 `aftershock_times`를 참조하도록 갱신하고, 변경된 횟수 한계값(15~25)을 검증하도록 수정 완료.

- [2026-06-29] Cache target_zones using tz_cpu to remove sync locks in nested loops.

## Phase 1E: Delta-based Aftershock Strike & Target KIA (2026-06-30)

### 변경 개요
단순 누적 확률 파괴 모델을 폐기하고, 여진 발생 시 Delta 충격량에 따라 에이전트와 타겟이 즉사하는 하드코어 물리 엔진 적용.

### 핵심 로직
- `affected_nodes` Dict 검사: `affected_nodes[c_node] >= 0.3` 시 UGV 100% 파괴.
- Target KIA (미션 증발): `affected_nodes[t_node] >= 0.5` 시 타겟 건물이 완전히 붕괴된 것으로 간주하여 `target_failed[b, i] = True` 처리.

## Phase 2G: Tensor Information Compression (Max Mapping) (2026-06-30)

### 변경 개요
Manager가 존의 위험도를 파악할 때 `zone_avg_node_dmg` 대신 최댓값을 매핑하여 극단적인 위험을 피하도록 유도.

### 핵심 로직
- 기존 파이썬 행렬 나누기 연산을 `scatter_reduce_(amax)`로 교체. (PyTorch 네이티브 연산으로 O(1) 복잡도 달성)
