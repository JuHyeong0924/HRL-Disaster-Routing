# `src/envs/worker_env.py` 로그

## 상세 코드 설명
- **역할**: 하위 Worker(Local Policy)를 위한 라우팅 및 물리 충돌 판정 환경입니다. 
- **주요 변경 로직**:
  - `step_batch` 내부에서 선택된 다음 노드(`action_idx`)에 도달하기 위해 거쳐야 할 간선(Edge) 정보를 점검합니다.
  - NetworkX 기반의 원래 맵 그래프(`self.G`)에 해당 간선이 남아있는지 `has_edge()`를 통해 확인하고, 만약 존재하지 않는다면 여진/지진에 의해 붕괴(Damage > 0.8 제거)된 것이므로 UGV의 진입 실패를 선언합니다.
- **텐서 논리**:
  - Batch 처리를 위해 `curr_idx`와 `action_idx`가 모두 `[B]` 형태의 텐서 연산을 통과하지만, 맵 단절 등의 예외 충돌 상황은 물리 시뮬레이션 성격이 강하므로 파이썬 루프 내에서 개별적 `NetworkX` 노드 객체(문자열 ID) 참조로 충돌(`stagnation` 등)을 판별한 뒤 `step_dones[b]`를 업데이트합니다.

## 시행착오 (Trial & Error)
- 기존에는 `mask[b, curr_nodes[b]] = 1.0` (Stagnation 제자리걸음) 조건만 있었으나, 강제 파괴 로직의 부재로 인해 데드라인 초과 이외에 에피소드가 실패하는 경우가 적었습니다. UGV 파괴 판정 로직 추가를 통해 더욱 가혹한 현실적 재난 시뮬레이션을 달성했습니다.

## 최적화 단계: get_action_mask_batch 루프 최적화 (2026-06-24)
- **변경 로직**: `get_action_mask_batch()` 내부에서 현재 노드에서 이웃 노드를 탐색할 때 $O(N)$으로 전체 416개 노드를 대조 판별하던 비효율적인 이중 루프를 $O(\text{deg}(v))$ 인접 리스트 `self._adj_list[c_idx]` 상의 노드들만 점검하는 방식으로 전면 수정.
- **효과**: 불필요한 간선 유효성 판단 연산 횟수가 매 스텝 배치당 수천 회에서 수십 회로 격감하여 CPU 오버헤드 해소.

## 텐서화 및 완전 벡터화 (2026-06-25)
- **변경 로직**: 런타임(`step_batch`, `_get_state_batch`, `get_action_mask_batch` 등)에서 NetworkX의 파이썬 Dict 조회 구조를 완전히 제거. `__init__` 및 `sync_tensors_from_graph()`에서 GPU 메모리에 텐서(`_adj_matrix_tensor`, `_weight_matrix`, `_damage_matrix`, `_status_matrix`)로 구조를 캐싱하고, PyTorch 고급 인덱싱 문법만 사용하여 시뮬레이션 전이 연산을 $O(1)$ 배칭 행렬 연산으로 탈바꿈.
- **의도**: CPU와 GPU 간의 병목 현상 및 파이썬 Loop 오버헤드 원천 차단. `test_phase1_env.py` 혹은 `verify_optimizations.py`를 통해 모든 로직이 동일하게 작동함을 무결점 증명.
