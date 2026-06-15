# Ablation Study v7: Worker 거리 표현 & Zone 구조

## 1. 개요
- **목적**: Worker State의 거리 계산(hop vs Dijkstra), Zone 정보(binary vs ternary), Zone Graph 가중치(uniform vs euclidean) 변경이 경로 효율에 미치는 영향 분석.
- **실행일**: 2026-06-15
- **환경**: `WorkerEnv` (Anaheim, Zone K=15, Grid 분할)
- **알고리즘**: PPO (4 epochs) + GAE + Entropy
- **고정 설정**: `soft_curr_next` + `use_pbrs` + `use_is_visited` + `use_relative_hop`
- **학습**: 5,000 episodes (batch=32, 156 steps)
- **평가**: 200 episodes, greedy(argmax), 고정 시드(42)
- **GPU**: NVIDIA RTX 4090 × 2

## 2. 변인 설명

| 변인 | 값 | 설명 |
|:---|:---|:---|
| `dist_mode` | `hop` | BFS 기반 unweighted 최단 홉 수. |
|  | `dijkstra` | 엣지 `weight` 기반 Dijkstra 최단 거리. **v7.1에서 `log1p` 변환 적용.** |
| `zone_info_mode` | `binary` | Ch.2 = {0, 1}. 다음 Zone 노드만 1. |
|  | `ternary` | Ch.2 = {-1, 0, 1}. 금지(-1), 현재(0), 목표(1). |
| `zone_weight_mode` | `uniform` | Zone 엣지 가중치 = 1.0. |
|  | `euclidean` | Zone 엣지 가중치 = 중심점 간 유클리디안 거리. |

## 3. v7 → v7.1 수정 사항

> [!IMPORTANT]
> v7 초기 실험에서 Dijkstra 거리를 `max_dist`로 정규화했을 때, 인접 노드 간 gradient가 hop 대비 2~3배 작은 **smoothing 문제**가 발견됨. v7.1에서 `log1p` 변환으로 해결.

| 항목 | v7 (초기) | v7.1 (수정) |
|:---|:---|:---|
| Dijkstra 정규화 | `dist / max_dist` (범위 [0, 83800]) | `log1p(dist) / log1p(max_dist)` (범위 [0, 11.3]) |
| Relative scale | `max_dist * 0.1 = 8380` | `3.0` (log 스케일 적합) |
| PBRS scale | `0.5 / (max_dist * 0.1)` | `0.5` (hop과 동일) |
| 평가 지표 | Hop Ratio만 | **Hop Ratio + Distance Ratio** |

v7.1 수정으로 Dijkstra 모드 B의 R_hop이 **1.201 → 1.157**로 개선됨 (smoothing 해결 확인).

## 4. v7.1 실험 설정

| ID | dist_mode | zone_info | zone_weight | 비고 |
|:---|:---:|:---:|:---:|:---|
| C | hop | ternary | uniform | v7 원본 |
| D | hop | binary | euclidean | v7 원본 |
| **Bv2** | dijkstra(log1p) | binary | uniform | v7.1 재학습 |
| **Ev2** | dijkstra(log1p) | ternary | euclidean | v7.1 재학습 |
| **F** | hop | ternary | euclidean | v7.1 신규 (최종 후보) |

## 5. 평가 결과

### 이중 지표 (Hop Ratio + Distance Ratio)

| ID | dist | zone | zw | SR | AvgPL | SP_h | **R_hop** | TotalDist | SP_d | **R_dist** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bv2** | dijkstra | binary | uniform | **100%** | 12.4 | 10.7 | 1.157 | **33174** | 28318 | **1.171** |
| C | hop | ternary | uniform | **100%** | 12.0 | 10.7 | **1.114** | 34099 | 28318 | 1.207 |
| D | hop | binary | euclidean | 100% | 11.9 | 10.7 | **1.110** | 34453 | 28392 | 1.205 |
| Ev2 | dijkstra | ternary | euclidean | 100% | 13.2 | 10.6 | 1.216 | 34274 | 28249 | 1.196 |
| F | hop | ternary | euclidean | **100%** | 12.4 | 10.7 | 1.148 | 34879 | 28318 | 1.225 |

- **R_hop**: Agent 경로 홉 수 / 최단 홉 수 (1.0 = 최적)
- **R_dist**: Agent 이동 물리 거리 / Dijkstra 최단 거리 (1.0 = 최적)

## 6. 핵심 분석

### 6.1 평가 지표에 따라 최선이 다름

| 기준 | 최선 모델 | R 값 | 해석 |
|:---|:---|:---:|:---|
| **최소 홉** (R_hop) | D (hop+binary+euclidean) | 1.110 | 최단 홉 대비 11% 추가 이동 |
| **최소 거리** (R_dist) | Bv2 (dijkstra+binary+uniform) | 1.171 | 최단 거리 대비 17% 추가 이동 |

### 6.2 hop vs dijkstra 근본적 차이

- **hop 모델** (C, D, F): 홉 수를 최소화하도록 학습 → 적은 스텝으로 도착하지만, 엣지 길이를 무시하여 **물리 거리가 더 길어짐** (R_dist ≈ 1.20~1.23)
- **dijkstra 모델** (Bv2): 물리 거리를 인식하여 학습 → 스텝은 약간 더 쓰지만 **물리 거리가 가장 짧음** (R_dist = 1.171)

### 6.3 R_dist 개선 한계

> [!WARNING]
> 전체적으로 R_dist가 1.17~1.23 범위로, 최적 대비 **17~23% 추가 이동**이 발생. 이는 현재 보상 구조의 한계:
> - `STEP_PENALTY = -0.1` (고정값) → 긴 엣지든 짧은 엣지든 동일 페널티
> - PBRS도 목적지까지의 거리 차이만 반영, 실제 이동한 엣지 길이를 페널티에 미반영
>
> 근본적 개선을 위해서는 `STEP_PENALTY`를 엣지 길이에 비례하게 하거나, 보상에 이동 거리 항을 추가해야 하지만 이는 Worker의 학습 안정성에 영향을 줄 수 있으므로 추후 검토.

### 6.4 log1p 효과 검증

| 실험 | v7 R_hop | v7.1 R_hop | 개선 |
|:---|:---:|:---:|:---:|
| B (dijkstra+binary+uniform) | 1.201 | 1.157 | **-0.044 ✅** |
| E (dijkstra+ternary+euclidean) | 1.192 | 1.216 | +0.024 (분산) |

B에서는 명확한 개선이 확인되었으나, E에서는 효과가 제한적. Dijkstra + ternary 조합은 ternary의 3값(-1/0/1) 정보와 log1p 거리 정보가 채널 2~3에서 간섭할 가능성이 있음.

## 7. 최종 채택

재난 라우팅의 실제 목표(물리적 거리 최소화)를 고려하여:

| 항목 | 확정 값 | 근거 |
|:---|:---|:---|
| **dist_mode** | `dijkstra` (log1p) | R_dist=1.171 (물리 거리 최소) |
| **zone_info_mode** | `binary` | Bv2가 dijkstra와 best 조합 |
| **zone_weight_mode** | `uniform` | Bv2가 dijkstra와 best 조합 |

> [!NOTE]
> **최종 선택: `dijkstra(log1p) + binary + uniform` (Bv2)**
> - SR=100%, R_hop=1.157, R_dist=1.171
> - 물리적 거리 기준 가장 효율적인 경로 생성
> - R_dist ≈ 1.17은 현재 보상 구조의 한계이며, HRL Manager 도입 시 거시적 경로 최적화로 추가 개선 기대
