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
