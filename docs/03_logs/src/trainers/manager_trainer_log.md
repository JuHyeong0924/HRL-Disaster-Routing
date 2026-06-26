# manager_trainer.py — 변경 로그

## Phase 2C: 보상 재설계 (2026-06-24)

### 변경 개요
HAZUS 동적 환경에 맞는 보상 shaping 적용.

### 보상 구성 요소
| 구성 요소 | 기존 | 신규 | 변경 근거 |
|----------|------|------|----------|
| 구출 보상 | ×10.0 | ×20.0 | 스파스 보상 중요도 증대 |
| PBRS 계수 | ×5.0 | ×2.0 | 밀집 보상이 스파스 보상을 압도하던 문제 해결 |
| 데드라인 만료 | 없음 | -5.0/target | 시간 긴급도 인식 강화 |
| 여유 시간 보너스 | 없음 | min(slack/max_time×5.0, 3.0) | 여유 있는 구출 장려 |

### 데드라인 만료 판정 로직
```python
for ti in range(num_targets):
    if (not target_rescued[b, ti] and
        not prev_target_failed[b][ti] and
        target_failed[b, ti]):
        turn_reward -= 5.0
```
- `prev_target_failed`를 턴 시작 시 스냅샷하여, 이번 턴에서 새로 만료된 타겟만 페널티 부여

## Phase 2F: Context Generator 호출부 업데이트 (2026-06-24)

### 변경 개요
- 추론 루프: `h_last` 제거, `num_feasible`/`avg_urgency` 실시간 계산
- PPO 미니배치 buffer: `h_last` 키 → `num_feasible`, `avg_urgency` 키로 교체
- `target_features.view(-1, 4)` → `view(-1, 6)` 전부 수정

### 호출부 변경
```python
# 기존
query = self.manager.generate_context(h_last, elapsed, rescued)
# 신규
num_feasible = target_mask.sum(dim=-1, keepdim=True).float() / num_targets
avg_urgency = (urgency_ch * mask_float).sum(-1, keepdim=True) / mask_float.sum(-1, keepdim=True).clamp(min=1)
query = self.manager.generate_context(elapsed, rescued, num_feasible, avg_urgency)
```

### Trial & Error
- **오류 없음**: 모든 구현이 첫 시도에 통과
