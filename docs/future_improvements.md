# 향후 개선사항 (Future Improvements)

> **목적**: HRL-Disaster-Routing 시뮬레이션의 **현실성(Fidelity)**과 **난이도(Challenge)**를 높이기 위한 구현 후보 기능 목록.
> 각 항목은 현재 시뮬레이션에서 빠져 있거나, 단순화되어 있는 부분을 식별하고 해결 방안을 제시함.

---

## 1. 🔴 노드 재난 (Node-level Disaster) — 우선순위: 높음

### 현재 문제
- 현재 재난(`apply_disaster_damage`)은 **간선(Edge)에만** 적용됨.
- 실제 지진에서는 **노드(교차로, 건물, 구조물)**도 붕괴/파손되어 통과 불가하거나 위험해짐.
- 노드가 파괴되면 해당 노드를 경유하는 **모든 경로**가 동시에 영향을 받으므로, 간선 단독 파괴보다 훨씬 치명적.

### 구현 방안
- `disaster_map.py`에 **Node Damage** 속성 추가 (`node_damage: float [0, 1]`).
- Node Damage 등급별 효과:
  | Damage 범위 | 상태 | 효과 |
  |-----------|------|------|
  | 0.0~0.3 | Normal | 통과 가능, 페널티 없음 |
  | 0.3~0.6 | Blocked | 통과 시 시간 페널티 (+15.0) |
  | 0.6~0.9 | Hazardous | 통과 시 UGV 파괴 확률 20%, 시간 페널티 (+30.0) |
  | 0.9~1.0 | Collapsed | **통과 불가** (해당 노드로의 이동 마스킹) |
- Worker의 State에 `node_damage` 채널 추가 (`[N, 1]` → 기존 State에 Concat).
- Manager의 Zone Features에 `zone_avg_node_damage` 채널 추가.

---

## 2. 🔴 동적 데드라인 (Dynamic Deadline / Time-Window Shrink) — 우선순위: 높음

### 현재 문제
- 타겟의 데드라인(TW)은 `reset()` 시 **한 번 고정**되면 에피소드 내내 변하지 않음.
- 실제 재난에서는 구조 대상자 근처에 추가 붕괴/여진이 발생하면 **생존 가능 시간이 급감**함.

### 구현 방안
- **여진(Aftershock) 발생 시**, 해당 여진이 타격한 Zone 내에 위치한 미구출 타겟들의 데드라인을 **동적으로 감소**:
  ```python
  # hrl_env.py 여진 처리 블록 내부
  for b in range(B):
      for i in range(self.num_targets):
          if not self.target_rescued[b, i] and not self.target_failed[b, i]:
              tgt_zone = self.target_zones[b, i].item()
              if tgt_zone in affected_zones:
                  # 잔여 데드라인의 20~40%를 감소
                  reduction = self.deadlines[b, i] * random.uniform(0.2, 0.4)
                  self.deadlines[b, i] = max(self.current_time[b].item() + 5.0,
                                             self.deadlines[b, i] - reduction)
  ```
- Manager의 Target Features에 이미 `time_remaining`과 `urgency_ratio`가 포함되어 있으므로, 데드라인이 변하면 이 피처들이 자동으로 반영됨 → **별도 State 수정 불필요**.
- **효과**: Manager가 "여진 발생 → 우선순위 재계산 → 긴급 타겟 먼저"라는 **반응형 의사결정**을 학습하게 됨.

---

## 3. 🟡 부분 관측 (Partial Observability / Fog of War) — 우선순위: 중간

### 현재 문제
- Worker와 Manager 모두 **전체 맵의 재난 상태를 완벽히 관찰** 가능 (Full Observability).
- 실제 재난 상황에서는 탐사하지 않은 구역의 도로 상태를 알 수 없음.

### 구현 방안
- **Vision Range** 도입: Worker가 현재 노드에서 `k`-hop 이내의 노드/간선 상태만 관측 가능.
- 미관측 간선의 damage 채널 = `0.5` (불확실성 기본값)로 설정.
- Manager는 Worker가 방문한 Zone만 정확한 `zone_disaster_intensity`를 받고, 미방문 Zone은 사전 확률 값 사용.
- **효과**: 탐색(Exploration) vs 활용(Exploitation) 트레이드오프 학습 유도.

---

## 4. 🟡 다중 UGV (Multi-Agent) — 우선순위: 중간

### 현재 문제
- 현재 시스템은 **단일 UGV**만 운용함.
- 실제 재난에서는 여러 대의 UGV가 동시에 다른 구역을 담당하여 구조 효율을 높임.

### 구현 방안
- `hrl_env.py`에 `num_agents` 파라미터 추가.
- Manager는 각 UGV에 대해 독립적으로 타겟을 할당하되, **이미 다른 UGV가 향하고 있는 타겟은 마스킹** 처리.
- 공유 보상(Team Reward) + 개별 보상(Individual Reward) 혼합 설계.
- **효과**: 협업/분업 전략 학습. 논문 기여도(Contribution) 대폭 향상.

---

## 5. 🟡 도로 복구 (Road Recovery / Repair Crew) — 우선순위: 중간

### 현재 문제
- 한번 파괴된 간선은 에피소드 종료 시까지 **영구적으로 손상** 상태.
- 실제 재난에서는 복구팀이 주요 도로를 점진적으로 복구함.

### 구현 방안
- 시간 경과에 따라 `damage`가 점진적으로 감소하는 **자연 복구** 로직 추가:
  ```python
  # 매 여진 체크 시 (또는 일정 시간 간격)
  for u, v in self.graph.edges():
      edge = self.graph[u][v]
      if edge['damage'] > 0:
          edge['damage'] = max(0.0, edge['damage'] - recovery_rate)
  ```
- 또는 Manager가 **"복구 명령"**을 액션으로 선택할 수 있게 확장 (Action Space 확장).
- **효과**: Manager가 "지금 우회할 것인가 vs 복구를 기다릴 것인가"의 시간적 트레이드오프를 학습.

---

## 6. 🟢 구조 대상자 상태 모델 (Victim Health Decay) — 우선순위: 낮음

### 현재 문제
- 타겟은 "데드라인 전 도착 = 성공 / 이후 = 실패"의 **이진(Binary)** 결과만 존재.
- 실제로는 빨리 갈수록 더 많은 생존자를 구할 수 있음 (연속적 보상).

### 구현 방안
- 타겟별 `health` 값 도입: 시간이 지남에 따라 선형/지수적으로 감소.
- 구출 시 보상 = `base_reward * remaining_health` (빨리 갈수록 높은 보상).
- **효과**: 보상 함수가 더 세밀해지고 (Dense), 시간 최적화 압력 강화.

---

## 7. 🟢 교통 혼잡 (Traffic Congestion) — 우선순위: 낮음

### 현재 문제
- 간선의 가중치는 재난에 의해서만 변함. 교통량에 의한 혼잡은 모델링되지 않음.

### 구현 방안
- 간선의 `capacity`를 활용한 BPR (Bureau of Public Roads) 함수:
  ```
  travel_time = base_time * (1 + α * (flow / capacity)^β)
  ```
- UGV 이동 시 해당 간선의 `flow` 증가, 이후 감소 (시뮬레이션).
- **효과**: 현실적인 네트워크 부하 모델링.

---

## 구현 우선순위 요약

| 순위 | 항목 | 난이도 | 영향도 | 비고 |
|:---:|------|:-----:|:-----:|------|
| 1 | 노드 재난 | ★★☆ | ★★★ | 간선 재난과 시너지, State 확장 필요 |
| 2 | 동적 데드라인 | ★☆☆ | ★★★ | 구현 간단, Manager 반응성 검증 핵심 |
| 3 | 부분 관측 | ★★★ | ★★★ | 대폭 리팩토링 필요, 논문 기여도 높음 |
| 4 | 다중 UGV | ★★★ | ★★★ | 완전 새 아키텍처 필요 |
| 5 | 도로 복구 | ★★☆ | ★★☆ | 시간적 트레이드오프 학습 |
| 6 | 대상자 상태 | ★☆☆ | ★★☆ | 보상 함수 세분화 |
| 7 | 교통 혼잡 | ★★☆ | ★☆☆ | 현실성 향상, 학습 난이도 큰 변화 없음 |
