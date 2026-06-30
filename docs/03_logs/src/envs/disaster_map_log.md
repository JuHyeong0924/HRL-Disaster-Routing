# disaster_map.py — 변경 로그

## Phase 1A: HAZUS Soft Closure (2026-06-24)

### 변경 개요
`apply_disaster_damage()` 메서드의 등급별 가중치 체계를 HAZUS Earthquake Model 기반으로 전면 재설계.

### 핵심 변경 사항
1. **간선 제거 코드 삭제**: `edges_to_remove` 리스트 및 `graph.remove_edge()` 루프 완전 제거
2. **가중치 배율 확대**: ×1.1/1.2/1.5 → ×1.0/2.0/4.0/20.0 (HAZUS Residual Capacity 역수)
3. **Complete 간선 Soft Closure**: `status='Closed'`, `weight = base_w * 20.0` 설정 (제거 대신)

### 설계 근거
- **FEMA HAZUS**: Residual Capacity (100%/50%/25%/0%) → Weight Multiplier = 1/RC
- **UGV 특수성**: Complete 등급에서 민간 차량은 통과 불가(0%)이나, UGV는 특수 구조 장비로 강행 돌파 가능. 대신 30% 파괴 확률 부여 (hrl_env.py에서 판정).
- **그래프 연결성**: `nx.is_connected(G)` 항상 보장 → Dijkstra 경로 탐색 실패 방지

### Trial & Error
- **오류 없음**: 첫 구현에서 8/8 테스트 통과

## Phase 1E: Delta-based Disaster Shock (2026-06-30)

### 핵심 변경 사항
- `apply_disaster_damage()`가 누적 파괴도 기반이 아닌, 순간 충격량(Delta) 딕셔너리(`{node_id: damage_increase}`)를 반환하도록 수정.
- 물리적 현실성을 위해 건물이나 에이전트의 파괴 기준을 누적 피해가 아닌 순간 가해진 충격의 크기로 변경함.
