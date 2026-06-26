# manager.py — 변경 로그

## Phase 2B: Manager 모델 구조 수정 (2026-06-24)

### 변경 개요
target_dim 4→6, context_proj 구조 변경 (h_last 제거)

### 핵심 변경 사항
1. **`__init__`**: `target_dim=4` → `target_dim=6`
   - `target_fusion[0]`: `Linear(hidden_dim+4, hidden_dim)` → `Linear(hidden_dim+6, hidden_dim)`
   - 예: `Linear(260, 256)` → `Linear(262, 256)`

2. **`context_proj`**: `Linear(hidden_dim+2, hidden_dim)` → `Linear(4, hidden_dim)`
   - 기존: `h_last(256) + elapsed(1) + rescued(1) = 258-dim`
   - 신규: `elapsed(1) + rescued(1) + num_feasible(1) + avg_urgency(1) = 4-dim`

3. **`generate_context()`**: 시그니처 변경
   - 기존: `generate_context(h_last, elapsed, rescued)`
   - 신규: `generate_context(elapsed, rescued, num_feasible, avg_urgency)`

### 설계 근거
- `h_last`는 항상 `torch.zeros()`로 초기화되어 실질적으로 0 벡터만 입력. 이는 `context_proj` 가중치 256개를 무의미한 0 입력에 할당하여 파라미터 낭비 + 노이즈 유발.
- 대체 피처 4개는 모두 [0, 1] 범위의 의미 있는 환경 상태를 인코딩.

### 체크포인트 호환성
- **비호환**: 기존 Manager 체크포인트는 사용 불가 (재학습 필요)
- **Worker 호환**: Worker 모델 구조 변경 없음

### Trial & Error
- **오류 없음**: 첫 구현에서 모든 텐서 흐름 테스트 통과
