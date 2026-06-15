# `src/envs/worker_env.py` Log

## 1. 코드 상세 설명
- **역할**: Worker 에이전트의 Phase 1 개별 라우팅 제어 환경.
- **핵심 로직 변경**:
  - `apply_dynamic_disaster(self)` 함수 파라미터에서 `accumulate` 불리언 인자를 제거하였습니다. 이는 `disaster_map`의 로직이 하드코딩된 누적 방식으로 개선됨에 따라 서명을 본래의 깔끔한 상태로 되돌리기 위함입니다.
  - 여전히 에피소드 진행 중(스텝 30, 스텝 60 등)에 함수가 호출되면 즉시 `disaster_map`을 통해 새로운 재난 피해를 누적 생성하고, `self._update_zone_graph_weights()`를 연쇄적으로 호출하여 상위 단위인 Zone Graph의 메타데이터 가중치도 즉시 동기화합니다.
  - **[신규] 시간 기반(Time-based) 페널티 개편**: `self.STEP_PENALTY (-0.1)`를 상수로 부과하던 로직을 전면 폐기하고, 물리적 이동 시간인 `edge_weight`에 비례하는 `time_penalty = -0.1 * edge_weight`를 매 노드 스텝마다 부과합니다. 이로써 워커는 파괴된 도로(고비용)를 피해 가장 '빠른' 경로를 개척하도록 강제 학습됩니다.
  - **[신규] 초고속 다익스트라 텐서 갱신 (`_update_dist_matrix`)**: `scipy.sparse.csgraph.shortest_path`를 도입하여 기존에 `O(V^3)`로 느렸던 플로이드-워셜 알고리즘을 밀리초(1ms 이하) 단위로 대체했습니다. 
  - **[신규] 시각 상실(Blindness) 버그 픽스**: 기존에는 초기 재난(`disaster_prob > 0`)이 발생해도 워커의 입력 텐서인 `self.dist_matrix`가 초기화 시점의 깨끗한 매트릭스로 영구 고정되는 치명적인 버그가 있었습니다. 이제 `reset()` 호출 시 재난이 터지면 즉각 `_update_dist_matrix()`를 통해 실시간으로 파괴된 다익스트라 맵을 신경망에 넘깁니다.
  - **[신규] 여진 발생 (`apply_aftershock`)**: 매니저의 지시를 덮어쓰지 않고 맵의 가중치 텐서만 기습적으로 파괴하는 새로운 여진 전용 메소드를 신설했습니다.

## 2. 알고리즘 & 텐서 로직
- 변경된 텐서 입출력은 없음. 기존의 `self.dist_matrix`는 가중치 기반 물리적 시간 거리를 캐싱하고, `self.hop_matrix` (존재할 시)는 칸 수를 담당함을 재확인.
- 워커의 상태 텐서(`is_curr`, `is_tgt`, `zone_info`, `dist`)에는 명시적 시계(`time`)가 제공되지 않으나, 다익스트라 최단 거리 텐서(`dist`)에 이미 소요 시간 정보가 모두 녹아있어 최적 경로를 판단할 수 있음.
- `self.max_dist`는 매 갱신 시마다 커지지 않도록 초기 클린 맵 기준값을 고정시켜, 신경망 정규화 스케일의 일관성을 완벽히 보장합니다.

## 3. Trial & Error (Troubleshooting)
- [Issue 1] `edge_weight` 기반 시간 패널티를 도입하면서 `edge_weight`가 0(Self-loop)인 노드에서의 무한 대기 현상 우려.
- [Fix 1] 워커 환경의 `action_idx == curr_idx`인 경우 `INVALID_PENALTY(-10.0)`와 함께 에피소드를 강제 종료하는 Stagnation 예외 처리 로직이 이미 견고하게 동작 중이므로 안전성을 확인.
- [Issue 2] 워커가 파괴된 도로를 보고도 돌진하는 현상 원인 규명.
- [Fix 2] `dist_matrix`가 한 번도 갱신되지 않았던 Blindness 버그를 확인하고 `Scipy`를 이용해 초고속 실시간 텐서 동기화 파이프라인으로 해결했습니다.
