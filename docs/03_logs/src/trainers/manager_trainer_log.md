# 📝 Manager Trainer Log (`src/trainers/manager_trainer.py`)

## 1. Intent & Purpose
*   **Role:** HRL Phase 2에서 Manager 네트워크를 PPO로 학습시키는 트레이너. Worker를 동결(Frozen)시킨 상태에서 매크로 스텝(타겟 선택 + Zone 지정)에 대한 Actor-Critic 업데이트를 수행함.
*   **Architecture:** 배치 단위 에피소드 수집(`_run_batch_episodes`) → GAE Advantage 계산(`_compute_gae`) → PPO 클리핑 업데이트(`train_step`).

## 2. Reward Design (Phase 2C)
| 항목 | 값 | 비고 |
|------|-----|------|
| 구출 보상 | +20.0 | 타겟 1개 구출 시 |
| 턴 패널티 | -0.5 | Manager turn 소모 |
| 시간 패널티 | -0.1 × elapsed | 물리적 소요 시간 비례 |
| PBRS | (log1p(prev) - log1p(curr)) × 2.0 | 거리 축소 보상 |
| 데드라인 만료 | -5.0 | 타겟별 |
| 여유 보너스 | min(slack/max_time × 5.0, 3.0) | 데드라인 내 여유 구출 시 |
| Zone 재방문 | -1.0 | Tabu 패널티 |
| 정체 | -2.0 | 같은 Zone 반복 + 구출 실패 시 |

## 3. Trial & Error (Debugging History)

### [2026-06-26] REWARD_SCALE = 0.1 도입
*   **증상**: Total Loss가 100~500 범위로 비정상적으로 높음. Value Loss(MSE)가 200~1000에 달하여 Critic의 거대한 기울기가 Policy Head의 학습을 압도.
*   **원인 분석**: 에피소드 누적 Return이 50~160 범위로 과도하게 큰 것이 근본 원인. MSE = (예측-Return)²이므로 자연스럽게 수백~천 단위로 폭등.
*   **수정**: `REWARD_SCALE = 0.1` 상수를 도입하여 PPO 내부에서는 축소된 보상(Return 5~16)을 사용. 통계 보고(`Rw`)에는 원본 스케일을 별도 추적(`ep_raw_rewards`)하여 인간의 해석 가능성 유지.
*   **결과**: V-Loss 200~1000 → **3.17**, Total Loss 100~500 → **1.57**로 안정화. Policy Loss는 기존과 동일하게 0.01~0.05 범위 유지.

### [2026-06-26] Batch Size 동적 기본값
*   Manager 학습 시 Worker가 동결되어 VRAM 여유가 충분하므로, `batch_size=256`, `mini_batch_size=2048`로 대폭 상향.
*   256개 에피소드를 1분 내 수집 완료하는 처리량 확보.
- [2026-06-29] Cache c_nodes, t_act, targets to CPU arrays to remove sync locks during reward calculations.

## 4. Reward Redesign (Phase 2H - Reward Anti-Hacking) [2026-06-30]
*   **증상**: HRL 모델이 Urgency Bonus(시간이 적게 남을수록 보상 증가)의 맹점을 악용하여 고의로 시간을 끄는 벼랑 끝 전술(Reward Hacking) 구사.
*   **조치**:
    1. **Base Rescue Reward 하향**: 중복 보상 방지를 위해 기본 구출 점수를 20.0에서 10.0으로 삭감.
    2. **Slack Bonus 도입**: `20.0 * (rem_time / tot_time)`. 빨리 구출할수록 최대 20.0의 보너스를 얻도록 로직 반전.
    3. **Global Rescue Rate Bonus**: 에피소드 종료 시 구출률에 비례하여 최대 50.0 보상 추가.
*   **결과**: 즉각적인 목표 구출을 우선시하는 정상적인 구조 우선순위 학습 성공.
