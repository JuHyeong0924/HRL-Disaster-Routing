# eval_utils.py — 변경 로그

## Phase 2F: Context Generator 호출부 + target_dim 업데이트 (2026-06-24)

### 변경 개요
`load_neural_models()` 및 `get_manager_action()`에서 Manager 모델 변경 사항 반영.

### 핵심 변경 사항
1. `Manager(target_dim=4)` → `Manager(target_dim=6)`
2. `tf = hrl_env.get_target_features().view(-1, 4)` → `.view(-1, 6)`
3. `elapsed` 정규화: `/ 100.0` → `/ max(hrl_env.max_time, 1.0)`
4. `h_last = torch.zeros(...)` 제거
5. `num_feasible`, `avg_urgency` 계산 및 `generate_context()` 호출 업데이트

### Trial & Error
- **오류 없음**
