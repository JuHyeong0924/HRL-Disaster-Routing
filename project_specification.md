# 재난 대응 UGV 경로 탐색 - 상세 설계서 (Low-Level Design)

**문서 버전:** 4.0
**최종 업데이트:** 2026-04-23 (v4.0: Worker State v3 개편 — x,y 제거 + edge 정규화 + time_pct 추가)
**작성 언어:** Korean (한국어)
**문서 설명:** 본 문서는 프로젝트의 모든 클래스, 함수 시그니처, 텐서 차원(Shape), 데이터 흐름을 코드를 보지 않고도 이해할 수 있도록 상세히 기술한 해설서입니다. 2026-04-07 기준 최신 RL 분리학습 구조, 성능 최적화, 버그 수정까지 포함합니다.

---

## 0. 현재 구현 상태 및 최근 진행 상황 (2026-04-07)

### 0.1 프로젝트의 현재 핵심 목표
본 프로젝트의 현재 최우선 목표는 다음 두 가지를 동시에 만족하는 **계층형 RL 경로 계획기**를 만드는 것입니다.

1.  **Manager**가 출발-도착 노드 쌍에 대해, 너무 짧지도 않고 너무 장황하지도 않은 **실행 가능한 sparse subgoal plan**을 생성할 것
2.  **Worker**가 해당 sparse plan을 실제 도로망 위에서 안정적으로 추종하고, 마지막 subgoal 이후에도 **goal까지 마무리(hand-off)** 할 것

초기 SL만으로는 기본적인 경로 모방이 가능했지만, RL fine-tuning 과정에서는 다음 병목이 반복적으로 나타났습니다.

*   Manager가 `1개 subgoal + EOS`로 지나치게 짧은 plan에 붕괴
*   corridor 안쪽으로는 가더라도, subgoal 배치 품질이 낮아 Worker가 정체(stagnation) 종료
*   Worker가 subgoal 근처까지는 가지만, 마지막 subgoal 이후 goal 전환이 약해 전체 성공률이 오르지 않음
*   Joint RL만으로는 Manager와 Worker가 서로의 노이즈를 증폭시키며 local optimum에 머무름

이 때문에 2026-03 후반부터 Phase 1 RL은 단일 trainer 기반 joint fine-tuning에서 벗어나, **Manager-only → Worker-only → readiness check → Joint(optional)** 구조로 재편되었습니다.

### 0.2 현재 RL 아키텍처 스냅샷
현재 RL 엔트리포인트는 [train_rl.py](/d:/연구실/연구/재난드론/Code/train_rl.py)이며, 다음 네 가지 실행 모드를 지원합니다.

*   `--stage manager`: Manager-only RL
*   `--stage worker`: Worker-only RL (Phase1GuidedWorkerTrainer 사용)
*   `--stage joint`: Joint RL (DOMOTrainer 사용)
*   `--stage phase1`: **`worker → manager → joint`를 자동 순차 실행** (각 stage에 동일 에피소드 수 할당)

즉, 현재 학습 파이프라인의 기준 checkpoint 흐름은 다음과 같습니다.

1.  **SL Pretrain**: `logs/sl_pretrain/model_sl_final.pt` (Worker + Manager)
2.  **RL Worker Stage**: `logs/rl_worker_stage/<timestamp>/final.pt` (Worker + Manager)
3.  **RL Manager Stage**: `logs/rl_manager_stage/<timestamp>/final.pt` (Worker + Manager)
4.  **RL Joint Stage**: `logs/rl_joint_stage/<timestamp>/final.pt` (Worker + Manager)
5.  **Phase 2 Disaster RL**: 향후 구현 예정

> **참고:** 모든 체크포인트에 `worker_state`와 `manager_state`가 함께 저장됩니다. (v3.8에서 `Phase1GuidedWorkerTrainer._save_worker_checkpoint`에 `manager_state` 추가)

### 0.3 2026-04-06 기준 반영된 주요 구조 변경
#### A. Worker 인터페이스 정상화
현재 [pomo_trainer.py](/d:/연구실/연구/재난드론/Code/src/trainers/pomo_trainer.py)의 rollout은 **stage-aware PBRS**를 사용합니다.

*   `worker-only`
    *   shaping target = `active subgoal if exists else goal`
    *   `lambda_sg = 1.0`, `lambda_goal = 0.0`
*   `joint/disaster` pre-handoff
    *   shaping target = `active subgoal`
    *   final-goal tether를 약하게 추가
    *   `lambda_sg = 1.0`, `lambda_goal = 0.2`
*   마지막 subgoal 이후
    *   `goal-only shaping`으로 자동 전환

또한 pointer가 바뀌는 step에서는 potential을 새 target 기준으로 rebase하여, subgoal 전환이 artificial reward spike를 만들지 않도록 했습니다.

#### A-2. Worker Critic 고도화 (2026-04-06)
기존 Worker Critic은 단일 `nn.Linear(hidden_dim, 1)`이었으나, 복잡한 PBRS 보상 환경에서 Explained Variance가 0 이하로 붕괴되는 문제가 발생했습니다.

현재 Critic 아키텍처는 **2-Layer MLP**로 교체되었습니다:
```python
self.critic = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1)
)
```

효과:
*   ExplVar: 초반 -0.03 → 후반 **0.29** (피크)까지 개선
*   기존 체크포인트 로드 시 `critic` 관련 키는 shape mismatch로 자동 무시되며, `train_rl.py`에 안전장치가 추가됨

#### B. Soft-arrival, adaptive skip, adaptive stagnation (2026-04-06 완화 강화)
Worker/Joint rollout은 이제 exact-arrival만 강제하지 않습니다.

*   `curr == sg`면 exact 도달
*   `hop(curr, sg) <= 1`이면 near 상태
*   near 상태이면서
    *   마지막 target이거나
    *   다음 target 기준으로 이미 더 좋아졌거나
    *   **다음 target 방향으로 접근 중(`approaching_next`: `hop(curr, next_target) < hop(prev, next_target)`)이면**
    *   `soft-arrival` 허용
*   target이 막혔거나 사실상 도달 불능이면 `adaptive skip` 허용

#### B-2. Post-Handoff Goal Regression 페널티 (2026-04-06 신규)
마지막 subgoal 이후 goal handoff 구간(`in_post_last_subgoal_phase`)에서 goal로부터 멀어지는 행동에 대해 **3.0배 강화된 페널티**를 적용합니다.

*   `goal_regression_penalty_large` (기본 0.35) × **3.0** = 실효 1.05
*   Worker가 goal 근처에서 불필요하게 방황하는 패턴을 억제
*   `best_goal_hops` 대비 현재 hop이 증가할 때 트리거

stagnation patience도 고정값이 아니라:

*   `clamp(10 + 2 * init_active_hops, 12, 40)`

로 바뀌었고, 아래 중 하나가 발생하면 카운터를 reset합니다.

*   active target hop 개선
*   goal distance 감소
*   pointer 변경
*   goal 도달
*   soft-arrival
*   skip

마지막 subgoal 이후 goal handoff가 일어나면 `stagnation_steps`, `best_subgoal_hops`, `segment_start_goal_dist`, `steps_since_last_subgoal`를 다시 초기화합니다.

#### D. Phase1 APTE 성능 최적화 (2026-04-06 v3.6)

Phase1 학습의 에피소드당 실행 시간을 **~2.4초 → ~0.6~0.8초 (3~4배 향상)**시키고, 배치 크기를 자동 최적화하는 변경입니다.

**D-1. GATv2 BPTT 차단 (`detach_spatial=True`)**

[phase1_guided_worker_trainer.py](/d:/연구실/연구/재난드론/Code/src/trainers/phase1_guided_worker_trainer.py) L449에서 `detach_spatial=False` → `True`로 변경.

*   Worker의 GATv2Conv 3-layer를 `torch.no_grad()` 내에서 실행 후 `.detach()`
*   **효과:** 400스텝 BPTT 연산 그래프 제거 → 속도 **3배** + VRAM **~60%** 절감
*   **트레이드오프:** GATv2(공간 인코더)는 SL 사전학습 가중치에서 동결됨. LSTM/Scorer/Critic은 gradient 유지
*   **근거:** SL 사전학습 Next-Hop 정확도 90%+로 충분히 수렴. RL에서의 추가 조정 이득 대비 속도/배치 이점이 압도적

**D-2. Checkpoint Hit 판정 벡터화**

기존: Python `for batch_idx in range(batch_size)` + `.item()` 호출 (GPU-CPU 동기화 N회)
변경: 체크포인트 리스트를 **2D 패딩 텐서(`ckpt_tensor`)로 사전 변환** 후, `batch_indices` 인덱싱으로 일괄 판정

```python
# 에피소드 시작 시 1회 변환
ckpt_tensor = torch.full((batch_size, max_ckpt), -1, device=device, dtype=torch.long)
# 매 스텝: 텐서 인덱싱으로 일괄 hit 판정
current_ckpt = ckpt_tensor[batch_indices, safe_ptrs]
hit_mask = ptr_valid & (next_nodes == current_ckpt)
```

*   **효과:** 에피소드당 ~20% 시간 절약

**D-3. GPU-CPU 동기화 최소화**

`path_traces` 기록 시 `.item()` N회 호출 → `.tolist()` 1회로 변경:
```python
_next_cpu = next_nodes.detach().cpu().tolist()
```

**D-4. 배치 크기 자동 탐색**

[train_rl.py](/d:/연구실/연구/재난드론/Code/train_rl.py)에 `_find_max_batch_size()` 함수 추가.

*   `--num_pomo auto` (기본값): VRAM 한도 내 최대 배치 크기를 자동 결정
*   탐색 후보: `[8, 12, 16, 24, 32, 48, 64]`
*   각 후보에 대해 5스텝 forward+backward 시뮬레이션 → OOM 발생 직전 크기 채택
*   **RTX 5080 실측:** `batch_size=24` 자동 채택 (기존 8의 **3배**)
*   명시적 지정도 가능: `--num_pomo 16`

**D-5. 에피소드 수 조정**

기본값: `--episodes 10000` → `5000`

*   근거: curriculum_ratio가 Ep 4000에서 1.0(최고 난이도) 도달, 이후 EMA 진동(수렴)
*   배치 3배 증가로 동일 rollout 수 달성: `5000 × 24 = 120,000 ≥ 10000 × 8 = 80,000`

**종합 예상 학습 시간:**

| 설정 | 기존 | 최적화 후 |
|------|------|----------|
| 배치 크기 | 8 | 24 (auto) |
| 에피소드 수 | 10,000 | 5,000 |
| 에피소드당 시간 | ~2.4초 | ~4초 (배치 3배) |
| Rollout/초 | 3.3/s | 6/s |
| **총 학습 시간** | **~6.7시간** | **~5.5시간 (rollout 1.5배)** |

#### E. Phase1 APTE 3대 버그 수정 (2026-04-07 v3.7)

`rl_debug_log.txt` 로그 분석 결과, POMO=64 학습에서 OOM 및 성능 정체(Stag 64%, ExplVar < 0, LR 고착)가 확인되어 아래 3가지 수정을 적용했습니다.

**E-1. LSTM 연산 그래프 단절 — VRAM 메모리 누수 방지**

*   **위치:** `_execute_goal_conditioned_rollout()` L453-454
*   **원인:** 에피소드를 일찍 종료한(active=False) 에이전트의 LSTM 상태(h, c)가 다음 스텝으로 전달될 때 `.detach()` 없이 넘어가, 불필요한 연산 그래프가 최대 400스텝 동안 VRAM에 누적
*   **수정:**
```python
# [수정 전] h = torch.where(active.unsqueeze(1), h_next, h)
# [수정 후] inactive 에이전트의 LSTM 상태는 detach
h = torch.where(active.unsqueeze(1), h_next, h.detach())
c = torch.where(active.unsqueeze(1), c_next, c.detach())
```
*   **효과:** 비활성 에이전트의 역전파 그래프 차단 → VRAM 누적 제거, OOM 위험 대폭 감소

**E-2. Advantage 정규화 추가 — Critic/Policy 학습 안정화**

*   **위치:** `_execute_goal_conditioned_rollout()` L673-679
*   **원인:** `pomo_trainer.py` 등 다른 트레이너에는 존재하는 Advantage Normalization이 Phase1 APTE 트레이너에만 누락. Policy Gradient가 방향성을 잃어 ExplVar가 음수(−0.059)로 무너짐
*   **수정:**
```python
advantages = flat_norm_returns - flat_values.detach()
# Advantage 정규화: 학습 안정화 및 Critic 설명력(ExplVar) 복구
if advantages.numel() > 1:
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
```
*   **효과:** 어드밴티지 스케일 정규화로 Policy Gradient 방향성 복구, ExplVar > 0 안정화 기대

**E-3. LR 스케줄러 하한선 버그 수정**

*   **위치:** `train()` 함수 L821-825
*   **원인:** `eta_min=min(config.lr, wkr_lr_floor)` — `min()`이 사용되어 `config.lr`이 `wkr_lr_floor`보다 작으면 하한선이 의미없이 더 낮은 값으로 설정됨. 실제 로그에서 Ep 0부터 LR이 1e-5로 바닥에 고착
*   **수정:**
```python
# [수정 전] eta_min=min(float(self.config.lr), self.wkr_lr_floor)
# [수정 후] max()로 변경 + config.lr의 10%를 후보에 추가하여 하한선 보장
eta_min=max(float(self.config.lr) * 0.1, self.wkr_lr_floor)
```
*   **효과:** `--lr 1e-4` 기준 `eta_min = max(1e-5, 1e-6) = 1e-5`로 정상 동작. CosineAnnealing이 1e-4 → 1e-5로 정상 감쇠

**로그 병목 진단 요약 (수정 전):**

| 지표 | Ep 0 | Ep 1600 | 판정 |
|------|------|---------|------|
| Wkr LR | 1e-5 (고착) | 1e-5 (고착) | ❌ 학습 동력 소실 |
| Stag Rate | 25.0% | 64.1% | ❌ 폭증 |
| ExplVar | −0.059 | −0.008 | ❌ Critic 붕괴 |
| SuccessEMA | 3.8% | 44.0% | ⚠️ 정체 |
| POMO | 64 | 64 | ❌ 권장 최대 24 |

**권장 재실행 명령어:**
```bash
python train_rl.py --stage phase1 --num_pomo 24 --lr 1e-4 --episodes 5000
```

#### C. Manager sparse planning / segment budget 교정
Phase 1의 Manager geometry는 이제 **weighted shortest path가 아니라 hop shortest path**를 기준으로 통일됩니다.

현재 반영된 핵심은 다음과 같습니다.

*   [disaster_env.py](/d:/연구실/연구/재난드론/Code/src/envs/disaster_env.py)에 `hop_next_hop_matrix`, `reconstruct_hop_shortest_path_indices()` 추가
*   reference anchor 생성은 hop shortest path 기반
*   `TARGET_SEGMENT_HOPS = 4.5`
*   `plan_len_ref = ceil(shortest_hops / 4.5)`
*   `plan_len_min = 1 if shortest_hops <= 4 else plan_len_ref`
*   `plan_len_max = plan_len_ref + 1`

또한 Manager는 단순히 “몇 개를 찍느냐”보다 “segment budget을 어떻게 나누느냐”를 직접 학습하도록 바뀌었습니다. 현재 plan diagnostics와 reward에 아래 항목이 추가되어 있습니다.

*   `segment_budget_error_mean`
*   `first_segment_budget_err_mean`
*   `first_segment_overshoot_mean`
*   `frontloaded_overshoot_rate`

즉, hard case에서 첫 subgoal이 너무 멀리 찍혀 뒤 token budget이 무너지는 failure mode를 직접 계측하고 억제하는 구조입니다.

#### D. Manager sparse warm-start (2026-04-18 재조정)
Manager-only stage는 학습 초반 **`40%`** episode 동안 sparse anchor target에 대한 CE 중심 warm-start를 수행합니다. (이전 `20%`에서 상향 조정 — SL 지식 보존 강화)

*   `sparse_warmstart_ratio = 0.40`
*   `warmstart_active=True`
*   `RL weight = 0.0`
*   `Aux CE weight = 1.0`

이후 나머지 `60%` 구간에서 RL + aux CE 혼합으로 전환합니다.

**RL 비중 스케줄 (2026-04-18 안정화):**

| 구간 | RL weight | Aux weight | 비고 |
|------|-----------|------------|------|
| 0~40% | 0.0 | 1.0 | A* 모방 전용 웜스타트 |
| 40~60% | 0.20 | 0.80 | RL 신호 소량 도입 |
| 60~80% | 0.35 | 0.65 | 점진적 RL 전환 |
| 80~100% | 0.50 | 0.50 | 최대 RL 비중 (이전 0.70에서 하향) |

**변경 근거:** 이전 실행(POMO=32)에서 RL 비중 0.70 도달 시 Plan Score가 -2.48→-10.64로 역행하고:Gradient Clip Hit 100%가 지속됨.

즉, dense/full-path SL prior에서 바로 sparse RL로 떨어지는 대신, sparse anchor sequence에 한번 더 맞춰주는 hybrid fine-tune subphase가 들어간 상태입니다.

#### E. Budget-aware decode + cone softening
Manager decode는 더 이상 고정된 radius/cone hard mask만 쓰지 않습니다.

*   `remaining_hops / remaining_slots` 기반 `seg_ref` 계산
*   `r_min = max(2, floor(0.5 * seg_ref))`
*   `r_max = ceil(1.5 * seg_ref) + 1`
*   `hop > r_max` 후보는 hard mask
*   `hop < r_min` 후보는 soft penalty
*   기존 135도 cone hard mask는 제거
*   뒤쪽 후보는 soft directional bias로만 억제

즉, Manager는 이제 “아무 3~10 hop 후보”가 아니라 “남은 budget에 맞는 segment 길이”를 중심으로 candidate를 평가합니다.

#### F. Worker robustness curriculum
Worker-only stage는 더 이상 clean reference sparse plan만 보지 않습니다.

현재 [worker_stage_trainer.py](/d:/연구실/연구/재난드론/Code/src/trainers/worker_stage_trainer.py)의 curriculum은 다음과 같습니다.

*   초반 `70%`
    *   `80% clean reference`
    *   `20% perturbed anchor`
*   후반 `30%`
    *   `50% clean reference`
    *   `30% perturbed anchor`
    *   `20% manager-sampled plan`

perturbation은 아래 세 종류만 허용합니다.

*   anchor 하나 drop
*   anchor rank `±1` 이동
*   첫 anchor 약간 overshoot

또한 마지막 subgoal 이후 goal이 active target이 된 직후 `10 step` 동안 worker aux CE에 **`5.0x`** handoff-aware boost를 줍니다 (Worker-only stage 기준; Joint에서는 `2.0x`). (이전 8 step / 1.5x에서 상향)

#### F-2. Worker 학습률 방어 (2026-04-06 신규)
Worker-only stage의 CosineAnnealing 스케줄러 `eta_min`에 하한선을 추가:

*   `eta_min = max(config.lr * 0.1, 1e-5)`
*   `--wkr_lr_floor` 인자로 외부에서 지정 가능 (기본 `1e-5`)
*   효과: 학습 후반부(Ep 8000+)에서도 학습 동력이 사라지지 않음

#### G. DOMOTrainer Adaptive DAgger & Micro Fine-Tuning (2026-04-18 신규)
Joint Stage 훈련의 고질적인 수렴 한계와 Manager의 엔트로피 폭주 현상을 방지하기 위해 `DOMOTrainer`에 두 가지 최적화가 추가되었습니다.

*   **Adaptive DAgger (EMA 쿨다운 스케줄링)**: Manager의 `entropy_ema`가 2.5를 넘거나, `clip_hit_ema`가 0.8을 돌파(+100 에피소드 이후)하면 일시적으로 5% 에피소드 분량의 쿨다운 타이머가 작동합니다. 이때 `aux_weight`가 평상시(0.15~0.20)에서 최대 0.90(1.0 - cooldown_ratio 비례)까지 상승하여 안정된 SL 지식 기반으로 폼을 교정하고 점차 RL 비중을 복원합니다.
*   **Worker 부분 동결 및 Micro Fine-Tuning**: Manager가 고정된 환경("움직이지 않는 과녁")에서 훈련할 수 있도록 전체 학습 에피소드의 첫 75% 동안 Worker 레이어의 그래디언트를 완전 동결(`requires_grad=False`)합니다. 진행도 75% 통과 시 잠금을 해제하고 Worker-specific Adam optimizer(`lr=1e-5`, CosineAnnealing)를 초기화하여, 마지막 25% 구간에 부드러운 End-to-End Joint 튜닝을 완수합니다. 이를 통해 연산 시간 단축과 학습 안정성이라는 두 마리 토끼를 잡았습니다.

##### G-1. 실행 루프 버그 수정 (2026-04-19)

1. **Bug #1 — 비활성 Worker 몽유병 (CRITICAL)**: `execute_batch_plan` 내 `env.step()` 호출 시, `active_mask=False`인 Worker(Goal 도달/Stagnation 종료)의 행동을 마스킹하지 않아, 이미 목적지에 도착한 Worker가 랜덤하게 이탈하여 최종 `is_success` 판정에서 실패 처리되는 치명적 버그. `actions = torch.where(active_before_step, actions, self.env.current_node)` 추가로 수정.
2. **Bug #2~#5 — `active_mask` → `active_before_step` 일관성**: 보상 누적 블록(`log_probs_sum`, `entropy_sum`, `step_penalty`, `loop_penalty`, `exploration`, `milestone`)에서 `active_mask` 대신 `active_before_step`을 명시적으로 사용하여, 코드 재정렬 시 발생할 수 있는 silent bug를 예방. 현재 코드 구조에서는 L2251~L2497 사이 `active_mask == active_before_step`이므로 기능 변화 없음.
3. **Bug #6 — `disaster_env.py` Dead Code**: `env.step()` 메서드 내 도달 불가능한 중복 `return` 문 제거.

##### G-2. Manager-Only RL 3단계 개선 (2026-04-20)

`ManagerStageTrainer._compute_manager_plan_score()` 및 `train()` 루프에 아래 6가지 개선 적용:

1. **Step 1 — Advantage 이상치 클리핑**: `torch.clamp(plan_score, min=-2.0)` + `torch.clamp(adv, -3, 3)`을 도입하여 `r_empty(-5.0)` 등 극단적 이상치가 정규화를 오염하는 문제 해결. `mgr_max_grad_norm`을 5.0→10.0으로 완화.
2. **Step 1 — Entropy 보너스 개선**: 고정 `-0.02`에서 `−0.05 × (1.0 − progress)`로 변경하여, 초반 탐색 강화 + 후반 수렴 유도.
3. **Step 2-a — 워커 실현 가능성 프록시(r_feasibility)**: `actual_segment_hops`에서 5칸 초과분에 이차 페널티(`-0.1 × excess²`)를 부과하여, 워커가 Stagnation Fail을 일으키는 장거리 구간을 사전 차단.
4. **Step 2-b — 목적지 도달 양수 보너스(r_goal_reached)**: 마지막 서브골이 목적지에 근접(홉≤1: +5.0, 홉≤3: +2.0)하면 강한 양수 보상. 기존 페널티 위주(9개 중 7개 음수) 구조의 소극적 수렴 문제 해결.
5. **Step 3-a — POMO Self-Teaching**: 학습 50% 이후, Quality Gate(r_feasibility > -0.5 AND 목적지 3홉 이내)를 통과한 최고 플랜을 일부 POMO 샘플의 Aux CE 교사로 주입. A* 고정 앵커 과적합 방지.
6. **Step 3-b — Contrastive Ranking Loss**: 학습 50% 이후, POMO 48개 샘플의 상위/하위 25% log_prob 차이를 logsigmoid 손실로 변환. REINFORCE 고분산 완화.
7. **Step 3-c — Adaptive SL Schedule**: 고정 40% 웜스타트 대신, `progress > 0.15 AND aux_ce_loss < 0.5` 조건 충족 시 조기 RL 전환.

##### G-3. 투트랙 학습 개선 (2026-04-20)

###### 트랙 1: Worker 재건 — hop_dist 피처 + Dense Reward

1. **hop_dist 피처 (col 9)**: `disaster_env.py` reset() 및 `update_target_features()`에 서브골까지의 정규화된 홉 거리(`hop_matrix[target] / max_hops`, [0,1] 범위) 추가. Env 피처: 9→10채널, Worker 입력: 8→9채널 (visit 제외).
2. **Worker node_dim 8→9**: `worker.py`, `train_sl.py`, `train_rl.py`, `eval_core.py`, `evaluate.py` 전체 동기화.
3. **Hop-based Dense Reward**: `get_reward()`에 `0.1 × (prev_hops − curr_hops)` Potential-based Shaping 추가. 기존 km 거리 셰이핑과 병행. inf 값 방어 포함.
4. **레거시 체크포인트 적응**: `_load_state_compat()`에서 7dim/8dim 입력 가중치를 9dim으로 자동 제로패딩 적응.

###### 트랙 1-B: Worker State v3 개편 (2026-04-23)

**목적:** cross-map 일반화 지원 + MDP 완전 관측성 보장 + GATv2 attention 편향 방지

1. **x,y 절대좌표 제거 (node_dim 9→8)**: 맵 암기(Memorization) 방지. 공간 인식은 `dir_x/y + net_dist`로 충분.
2. **time_pct 피처 추가 (col 7)**: `current_step / max_steps` [0,1]. 에이전트가 마감 시간을 인지하여 타임아웃 방지. 추후 SOC(배터리 잔량)로 대체 예정.
3. **Edge Feature Min-Max 정규화**: `[length, capacity, speed]`의 스케일 불일치 해소. Per-graph Min-Max로 cross-map 호환.

**변경 후 Worker 입력 구조:**
```
Node: [is_curr, is_tgt, net_dist, dir_x, dir_y, is_final, hop_dist, time_pct] = 8채널
Edge: [length, capacity, speed] (Min-Max 정규화) = 3채널
```

**정규화 현황:**
| 피처 | 범위 | 방법 |
|------|------|------|
| is_curr, is_tgt, is_final | {0, 1} | one-hot/binary |
| net_dist | [0, 1] | `apsp / max_dist` (그래프 내 최대) |
| dir_x, dir_y | [-1, 1] | 단위벡터 |
| hop_dist | [0, 1] | `hops / max_hops` |
| time_pct | [0, 1] | `step / max_steps` |
| edge features | [0, 1] | Per-graph Min-Max |

| 변경 파일 | 핵심 수정 |
|---|---|
| `worker.py` | `node_dim=8` 기본값, v3 docstring |
| `train_sl.py` | `WorkerLSTM(node_dim=8)`, worker_in 8채널 (x,y 제거), edge 정규화 |
| `train_rl.py` | `WorkerLSTM(node_dim=8)` |
| `worker_nav_trainer.py` | `_build_worker_input(time_pct)`, `_select_edge_attr()` 정규화, `detach_spatial` config화 |
| `eval_core.py` | node_dim=8, `build_worker_input(time_pct)`, `select_edge_attr()` 정규화 |
| `evaluate.py` | node_dim=8 |

###### 트랙 2: Manager Soft Diet + Hard Mining

1. **보상 Soft Diet**: `_compute_manager_plan_score()`에서 `plan_score = r_main + r_soft + r_fatal`.
   - **메인 (가중치 1.0)**: `r_goal_reached + r_corr + r_spacing`
   - **Soft 가드레일 (가중치 0.1)**: `r_count + r_anchor + r_mono + r_budget + r_front + r_first`
   - **Fatal (가중치 1.0)**: `r_empty`
2. **Hard Example Mining**: `__init__`에 `deque(maxlen=2048)` 버퍼. `plan_score.max() < 0.0`인 OD 쌍 수집, 25% 확률로 재출제. `env.reset(forced_start=, forced_target=)` 지원.
3. **Manager-Only RL 에피소드**: 20,000 에피소드 (Worker 학습 대기 시간 활용).

#### H. Readiness gate와 artifact
Phase 1 pipeline은 이제 무조건 joint로 가지 않습니다.

`manager -> worker`가 끝나면 [train_rl.py](/d:/연구실/연구/재난드론/Code/train_rl.py)가 다음 artifact를 생성합니다.

*   `logs/rl_phase1_joint/stage_readiness.json`
*   `logs/rl_phase1_joint/val_by_refbin.csv`

그리고 readiness를 계산합니다.

*   `smoke_ready`
*   `launch_ready`

`launch_ready=False`이면 `--stage phase1`은 joint를 자동으로 건너뜁니다. 다만 연구용으로는 `--force_joint`로 gate를 우회할 수 있습니다.

주의:

*   현재 `val_by_refbin.csv`는 **최신 stage debug row를 ref-bin 형식으로 요약한 lightweight artifact**입니다.
*   엄밀한 held-out fixed validation sweep은 아직 별도 evaluator로 구현되어 있지 않습니다.

#### H. Runtime 안정화 패치
다음 hygiene patch도 이미 반영되어 있습니다.

*   value loss를 alive/valid step 기준으로 평균
*   `loss.requires_grad=False`인 degenerate batch는 optimizer step skip
*   각 stage save dir에 `runtime_config.json` 저장
*   manager/worker/joint 모두 최신 debug metrics를 CSV/JSONL로 기록

#### I. 2026-04-18 수정 사항 요약

**I-1. Worker 시각화 버그 수정**

*   **위치:** `src/trainers/worker_nav_trainer.py` L867
*   **문제:** `success_ema`가 % 단위(0~100)로 저장되는데, 부모 클래스 `DOMOTrainer._plot_rl_curves`가 이를 0~1 비율로 가정하고 `× 100.0`을 곱해 그래프가 항상 0% 또는 범위 초과
*   **수정:** `success_ema / 100.0`으로 변환하여 부모 클래스 로직과 호환
*   **효과:** RL 학습 곡선에서 성공률이 정상 표시됨

**I-2. POMO 기본 배치 크기 변경 (32 → 48)**

*   **위치:** `train_rl.py` L203
*   **수정:** `args.num_pomo` fallback 값을 32 → **48**으로 변경
*   **근거:** 24GB VRAM GPU에서 Worker 단독 ~13GB → POMO=48은 ~19GB로 안전 범위
*   **효과:** Baseline 정확도 향상으로 gradient 안정화 + 탐색 다양성 증가

**I-3. Manager Stage 학습 안정화** (상세: §D 참조)

이전 실행(POMO=32) 분석 결과에 기반한 7개 하이퍼파라미터 수정:
*   `mgr_max_grad_norm`: 20.0 → 5.0 (gradient 폭발 차단)
*   `sparse_warmstart_ratio`: 0.20 → 0.40 (SL 지식 보존)
*   LR `hold_ratio`: 0.7 → 0.4 (LR 조기 감소)
*   LR `min_factor`: 0.3 → 0.1 (후반 미세 조정)
*   RL 최대 비중: 0.70 → 0.50 (역행 방지)
*   Empty plan 페널티: -2.0 → -5.0 (겁쟁이 전략 차단)
*   적정 범위 보너스: +0.50 신규 (보상 해킹 방지)


### 0.4 2026-04-06 리팩토링 및 학습 결과

#### 0.4.1 리팩토링 변경 요약 (2026-04-06)
기존 40% 성공률 Plateau를 돌파하기 위해 아래 5가지 구조적 변경을 적용했습니다.

| 파일 | 변경 내용 | 효과 |
|------|----------|------|
| `worker.py` | Critic `nn.Linear` → **2-Layer MLP** (`Linear-ReLU-Linear`) | ExplVar 붕괴 방지 → 0.29 달성 |
| `pomo_trainer.py` | Soft-Arrival에 `approaching_next` 조건 추가 | 불필요한 정체 감소 |
| `pomo_trainer.py` | Post-Handoff Goal Regression 페널티 **3.0x** | goal 근처 방황 억제 |
| `manager_stage_trainer.py` | `warmstart_ratio` 0.35 → **0.20** | 초기 CE 웜스타트 구간 조정 |
| `worker_stage_trainer.py` | `handoff_aux_boost` 2.0→**5.0**, `eta_min`→**`max(lr*0.1, 1e-5)`** | Handoff 학습 강화 + 후반부 학습 동력 보장 |
| `train_rl.py` | `--stage` 멀티스테이지 라우팅 + `--wkr_lr_floor` + Critic 체크포인트 안전장치 | 유연한 커리큘럼 실행 + 호환성 |

#### 0.4.2 Phase 1 APTE 학습 최종 결과 (Ep 10000/10000, EMA 89.3%)
`python train_rl.py --stage phase1 --num_pomo 32 --lr 1e-4 --episodes 10000 --debug` 완전 실행 결과:

##### EMA 추이
| 구간 | 에피소드 | SuccessEMA | curriculum_ratio | 난이도 |
|------|---------|-----------|-----------------|--------|
| 초반 | 1000 | 58.9% | 0.125 (쉬움) | 0~20% 거리 |
| 중반 | 3000 | 53.3% | 0.375 (보통) | 30~50% 거리 |
| 난이도 상승 | 5000 | 49.0% | 0.625 (어려움) | 50~70% 거리 |
| 고난이도 | 7000 | 59.0% | 0.875 (준최고) | 70~90% 거리 |
| 피크 회복 | 8200 | 91.1% | 1.0 (최고) | 80~100% 거리 |
| **최종 완료** | **10000** | **89.3%** | **1.0 (최고)** | **80~100% 거리** |

##### 핵심 지표 분석 (최종 10,000 에피소드 기준)
*   **Stagnation Fail**: 50~87% (초반) → **6.2%** (최종)
*   **Loop Fail**: 전 구간에서 **0%** (완전 해결)
*   **Explained Variance**: -0.03 (초반) → **0.318** (최종, 매우 안정적)
*   **AuxCE**: 4.83 → **2.72** (지속 감소, 전문가 경로 모방성 극대화)
*   **Hop1→Success**: **100%** 달성 (마무리 전환 완벽)
*   **Goal<=2 hop**: 최고 난이도에서도 **93~100%** 달성

##### 리팩토링 효과 판정
| 개선 항목 | 근거 | 판정 |
|----------|------|------|
| Critic 2-Layer MLP | ExplVar: -0.03 → 0.29 | ✅ 확인 |
| Soft-Arrival 완화 | 최고 난이도에서 Stag 12% | ✅ 확인 |
| Handoff 강화 | Hop1→Succ 85~100% | ✅ 확인 |
| eta_min 1e-5 | Ep 7800에서도 Loss 감소 지속 | ✅ 확인 |
| AuxCE 감소 | 4.83 → 3.17 | ✅ 확인 |

해석:
*   **40% Plateau를 완벽하게 돌파**하여 최고 난이도(80~100% 거리)에서 **EMA 89.3% (단일 에피소드 93.8%)** 달성.
*   Critic 네트워크의 2-Layer MLP 개편과 Handoff 페널티 스케일업이 병목 해결의 핵심(Key)으로 작용.
*   장거리 문제(TeacherHop 19~21)에서도 안정적인 추종 능력을 보임.

##### 잔여 한계점
*   GoalRegress(목표 이탈)가 여전히 40~60%대로 발생. 다만 이것이 실패로 이어지지는 않고 최종적으로는 성공(100%)하기 때문에 치명적인 결함은 아님.

### 0.5 현재 결론 (2026-04-08 업데이트)
현재 프로젝트의 **Phase 1 (APTE 커리큘럼)이 완벽하게 성공(EMA 89.3%)**했습니다.

*   **Worker**
    *   Critic 2-Layer MLP와 Handoff 강화 설계 덕분에 Stagnation(정체)과 Loop 문제를 완벽히 극복했습니다.
    *   목표 반경 1-hop에 도달하면 무조건(100%) 목표점을 찍는 강건한 마무리 능력을 갖췄습니다.
*   **다음 단계**
    *   이 강력한 사전학습 가중치(`logs/rl_phase1_apte/best.pt`)를 바탕으로, 본격적인 **Phase 2 (재난 상황 적응 - `--disaster` 옵션 적용)** 학습 또는 **완전한 Joint RL** 미세조정으로 넘어갈 준비가 완료되었습니다.

### 0.6 당장 다음에 확인해야 할 핵심 지표
Phase 1 재실행 시, 아래 지표를 우선적으로 봅니다.

*   success_ema — 전체 성능 EMA
*   stagnation_fail_rate — 주요 실패 원인
*   critic_explained_variance — Critic 학습 품질
*   goal_regression_after_best4_rate — 목표 이탈 빈도
*   goal_hop_1_to_success_rate — Last-Mile 변환율
*   hidden_checkpoint_hit_rate — 중간 체크포인트 통과율
*   worker_aux_ce_loss — 전문가 경로 모방 품질
*   path_length_ratio — 최적 경로 대비 실제 경로 비율

이 중 특히

*   stagnation_fail_rate < 20% + success_ema > 65%
    *   Worker 핵심 역량 달성 기준
*   critic_explained_variance > 0.2 (안정적)
    *   Critic이 Advantage를 정확히 추정하여 정책 개선을 가속하는 기준

으로 해석하는 것이 현재 코드와 가장 잘 맞습니다.

### 0.7 보상 함수 상세 설계 (2026-04-08 추가)

소스: `src/trainers/pomo_trainer.py` (`DOMOTrainer.execute_batch_plan()` + `_compute_plan_reward_adjustment()`)

#### 보상 구조 개요
최종 보상 = **Worker 기본 보상 (R1~R6, P1~P3)** + **Manager Plan Adjustment (R7)**

Worker와 Manager는 동일한 `final_reward`를 공유하되, Manager에게만 Plan 품질에 대한 `plan_adjustment(R7)`이 추가된다.

#### A. Worker 보상 (경로 실행 품질)

| 코드명 | 이름 | 수식/설명 | 하이퍼파라미터 |
|--------|------|-----------|--------------|
| **R1** | PBRS | `Σ[λ_sg × (γφ_new_sg - φ_old_sg) + λ_goal × (γφ_new_goal - φ_old_goal)] × SCALE` | POTENTIAL_SCALE=6.5, γ=0.99 |
| **R2** | Subgoal 도달 | `Σ[0.5 × (BASE + SCALE × progress) + OPT × ratio]` (eligible 조건 충족 시) | BASE=0.5, SCALE=1.5, OPT=0.5 |
| **R3** | Goal 도달 | `is_success × GOAL_REWARD` | **GOAL_REWARD=40.0** |
| **R4** | 효율성 보너스 | `clamp(MAX × (2 - actual/optimal), 0, MAX)` (성공 시에만) | EFFICIENCY_MAX=8.0 |
| **R5** | 마일스톤 | 25/50/75% 거리 통과 시 일회성 보너스 | +0.75, +1.5, +3.0 |
| **R6** | 탐색 보너스 | 신규 노드 방문 시 소액 (현재 비활성) | EXPLORATION_BONUS=**0.0** |
| **P1** | 시간 압박 | `Σ[-0.02 × (1 + 2.0 × t/T)]` (스텝이 길수록 강화) | BASE=-0.02, PRESSURE=2.0 |
| **P2** | 루프/퇴보 | 재방문: `-0.1 × max(count-1, 0)`, Goal 퇴보: `-3.0` (Handoff 후) | LOOP_SCALE=0.1, LIMIT=6 |
| **P3** | 실패 | `(!success) × FAIL_PENALTY` | **FAIL_PENALTY=-20.0** |

*   **Stage별 PBRS 차이**: Worker-only(`λ_sg=1, λ_goal=0`), Joint(`λ_sg=1, λ_goal=0.2`), Goal Finish 구간(`λ_goal × 2.0`)
*   **Subgoal 도달 조건**: 정확 도착 OR Soft Arrival(1-hop 이내 + 다음서브골접근/지나침/마지막서브골)

#### B. Manager Plan Adjustment (R7)

R7 = clamp(checkpoint_quality + plan_penalty, min=-6.0, max=+4.0)

*   **Checkpoint Quality** (최대 4.0): `0.6 × 효율성(경유거리/최적거리) + 0.4 × 균등분할(세그먼트분산)`
*   **Plan Penalty** (음수 합산):

| 구성 요소 | 가중치 | 설명 |
|---------|--------|------|
| Corridor 위반 | -1.5 × deficit | 서브골이 S→G 복도(+2hop) 밖이면 페널티 |
| Plan 개수 미달 | -1.0 × under | 기준 서브골 수 미달 시 |
| Plan 개수 초과 | -0.2 × over | 기준 초과 시 소량 |
| 간격 불균일 | -0.8 × err | APSP Anchor 대비 진행률 오차 |
| 비단조성 | -0.5 × viol | 서브골이 Goal에서 멀어지면 페널티 |
| Anchor 오차 | -0.15 × hop_err | A* 기준점 대비 hop 거리 |
| Anchor 근접 | +0.5 × ratio | 1-hop 이내 Anchor 비율 보너스 |
| 첫 Anchor | -0.3 × err | 첫 서브골 Anchor 정확도 가중 |
| 빈 Plan | -2.0 | 서브골 0개 시 고정 |

#### C. 보상 스케일 비교

| 보상 | 성공 시 | 실패 시 |
|------|--------|---------|
| R1 PBRS | +5 ~ +15 | -5 ~ +5 |
| R2 Subgoal | +1 ~ +5 | 0 ~ +2 |
| R3 Goal | **+40** | 0 |
| R4 Efficiency | 0 ~ +8 | 0 |
| R5 Milestone | +0.75 ~ +5.25 | 0 ~ +3 |
| P1 Time | -2 ~ -8 | -4 ~ -12 |
| P2 Loop | -0.5 ~ -3 | -2 ~ -10 |
| P3 Fail | 0 | **-20** |
| R7 Plan | -6 ~ +4 | -6 ~ +4 |
| **합계** | **+30 ~ +60** | **-40 ~ -10** |

---

## 1. 물리 계층 (Physical Layer)

### 1.1 `src/agents/robot.py`
RoboCue-X 로봇의 물리적 특성과 에너지 소모 공식을 정의합니다.

#### **Class `BaseRobot`**
*   **역할**: 로봇의 상태 관리 및 물리 엔진 코어
*   **속성 (Member Variables)**:
    *   `battery_j (float)`: 현재 배터리 잔량 (Joule). Max: 36,000,000 J (10 kWh)
    *   `base_speed (float)`: 기본 주행 속도 (40.0 km/h)
    *   `rated_efficiency_wh_per_km (float)`: 기준 전비 (50.0 Wh/km)

#### **Method `_calculate_physics`**
물리적 상호작용을 계산하는 핵심 함수입니다. (Scalar 연산)
```python
def _calculate_physics(self, length_km: float, status: str, damage: float) 
    -> (float, float, float, float):
```
*   **Args**:
    *   `length_km`: 엣지 길이
    *   `status`: 지형 상태 ('Normal', 'Caution', 'Danger', 'Closed')
    *   `damage`: 파괴도 (0.0 ~ 1.0)
*   **Logic (Pseudo-code) - Simplified**:
    1.  **Status Check**:
        *   **Closed**: Spd=0.0 (Cost=High)
        *   **Others**: Spd=Base_Speed (40km/h) (Cost=0)
    2.  **결과 산출**:
        *   `Real_Speed = Base_Speed`
        *   `Energy_J = Inf (Disabled)`
*   **Returns**: `(time_h, energy_percent, real_speed_kmh, energy_joule)` (Energy values are dummy)

*   **Returns**: `(time_h, energy_percent, real_speed_kmh, energy_joule)`

---

## 2. 데이터 처리 파이프라인 (Data Pipeline)

### **2.1. Graph Loader** (`src/utils/graph_loader.py`)
- **역할**: TNTP 형식의 맵 데이터를 파싱하여 NetworkX 및 PyG(Pytorch Geometric) 데이터 객체로 변환.
- **주요 기능**:
    - `_load_network`: 노드 좌표 및 엣지 정보 로딩.
    - `get_pyg_data`: PyG Data 객체 생성.
    - **[중요] 좌표 정규화 (Cross-Map Generalization)**:
        - 노드 좌표 $(x, y)$를 `[0, 1]` 범위로 Min-Max Normalization 수행.
        - **효과**: 맵의 실제 크기(km)가 달라도 모델은 동일한 상대적 거리감을 학습.
        - **확장성**: `Anaheim` 학습 모델을 `SiouxFalls` 등 다른 스케일의 맵에 즉시 적용 가능.

### 4.2 RL Fine-tuning Pipeline (2026-03-31 최신 구조)
**목적:** SL로 초기화된 계층형 정책을 실제 rollout에서 강화하되, Manager와 Worker가 서로의 노이즈 때문에 동시에 붕괴하는 문제를 피하기 위해 **Phase 1을 3단계로 분리**하여 학습합니다.

#### 1단계: Phase 1 Manager-only RL
*   **실행 명령어:** `python train_rl.py --stage manager --map [맵이름] --episodes [N]`
*   **환경 상태:** `enable_disaster=False`
*   **목적:** Worker를 완전히 제외하고, Manager가 **적정 길이 + 적정 spacing + segment budget + anchor 정렬**을 만족하는 sparse plan을 만들도록 학습합니다.
*   **초기 체크포인트:** `logs/sl_pretrain/model_sl_final.pt`
*   **세부 동작:**
    *   stage 초반 `20%`는 sparse anchor CE 중심 warm-start
    *   이후 `80%`는 RL plan score + auxiliary CE 혼합
    *   reference anchor는 Phase 1에서 **hop shortest path** 기준으로 생성
*   **출력 체크포인트:** `logs/rl_phase1_manager/best.pt`, `final.pt`

#### 2단계: Phase 1 Worker-only RL
*   **실행 명령어:** `python train_rl.py --stage worker --map [맵이름] --episodes [N]`
*   **환경 상태:** `enable_disaster=False`
*   **목적:** reference sparse plan을 기준으로 Worker의 path-following과 last-mile handoff 능력을 안정화합니다.
*   **초기 체크포인트:**
    *   Manager: `logs/rl_phase1_manager/best.pt`가 있으면 우선 사용, 없으면 `logs/sl_pretrain/model_sl_final.pt` fallback
    *   Worker: `logs/sl_pretrain/model_sl_final.pt`
*   **세부 동작:**
    *   초반 `70%`: `80% clean reference + 20% perturbed anchor`
    *   후반 `30%`: `50% clean reference + 30% perturbed anchor + 20% manager-sampled plan`
    *   `handoff-aware CE`를 사용하여 마지막 subgoal 이후 첫 `8` step을 더 강하게 감독
*   **출력 체크포인트:** `logs/rl_phase1_worker/best.pt`, `final.pt`

#### 3단계: Phase 1 Joint RL
*   **실행 명령어:** `python train_rl.py --stage joint --map [맵이름] --episodes [N]`
*   **환경 상태:** `enable_disaster=False`
*   **목적:** 앞서 학습된 Manager/Worker를 합쳐, end-to-end interaction 하에서 최종 성공률을 미세 조정합니다.
*   **초기 체크포인트:**
    *   Manager: `logs/rl_phase1_manager/best.pt`
    *   Worker: `logs/rl_phase1_worker/best.pt`
*   **세부 동작:**
    *   stage-aware PBRS와 soft-arrival/handoff logging을 유지
    *   manager/worker auxiliary loss는 약한 정렬용 regularizer로만 사용
    *   기본적으로 readiness gate를 통과해야 실행되며, 연구용 확인이 필요할 때만 `--force_joint`로 우회 가능
*   **출력 체크포인트:** `logs/rl_phase1_joint/best.pt`, `final.pt`

#### 4단계: Phase 1 전체 자동 실행
*   **실행 명령어:** `python train_rl.py --stage phase1 --map [맵이름] --episodes [N]`
*   **동작:** `manager -> worker -> readiness check -> joint(optional)`를 한 프로세스에서 순차 실행
*   **용도:** 사람이 stage를 수동으로 이어붙이지 않고, Phase 1 전체를 한 번에 돌리고 싶을 때 사용
*   **주의:** `--stage phase1`과 `--disaster`는 동시에 사용할 수 없습니다.
*   **추가 산출물:**
    *   `logs/rl_phase1_joint/stage_readiness.json`
    *   `logs/rl_phase1_joint/val_by_refbin.csv`
*   **Gate 동작:**
    *   `launch_ready=False`이면 자동 joint는 건너뜀
    *   `--force_joint`가 있으면 readiness fail이어도 joint를 강제로 실행

#### 5단계: Phase 2 Disaster RL
*   **실행 명령어:** `python train_rl.py --map [맵이름] --episodes [N] --disaster`
*   **환경 상태:** `enable_disaster=True`
*   **목적:** Phase 1 joint에서 확보한 기본 길찾기/플래닝 능력을 바탕으로, 재난으로 인해 edge 상태가 동적으로 바뀌는 상황에 적응하도록 fine-tuning합니다.
*   **초기 체크포인트:** `logs/rl_phase1_joint/best.pt`
*   **출력 체크포인트:** `logs/rl_finetune_phase2`

**핵심 아키텍처 요약**
*   Manager stage: sparse plan quality와 segment budget만 학습
*   Worker stage: reference/noisy sparse plan 추종과 handoff 안정화만 학습
*   Joint stage: interaction alignment만 수행
*   Disaster stage: dynamic routing adaptation만 수행

즉, 현재 설계는 “한 번에 모든 것을 RL로 해결”하는 구조가 아니라, **학습 안정성을 위해 역할을 단계별로 분리한 후 다시 합치는 구조**입니다.

---

## 3. 시스템 구동 흐름 (System Operational Flow)

본 시스템이 강화학습을 통해 최적 경로를 찾아내는 전체 과정은 다음과 같습니다.

### 3.1 통합 학습 파이프라인 (Integrated Training Pipeline - `train.py`) [NEW]
본 시스템은 단일 스크립트로 전체 학습 및 검증 과정을 수행할 수 있습니다.
*   **Command**: `python train.py --map Anaheim`
*   **Workflow**:
    1.  **Phase 1: Supervised Pre-training (`train_sl.py`)**:
        *   전문가 경로 데이터를 모방 학습.
        *   **DAgger 스타일 오류 복구 훈련**: APSP Next Hop 테이블로 모델 예측 위치에서의 정답 재계산.
        *   **TF Ratio**: `1.0 → 0.0` (에폭 10부터 완전 자율 추론 + DAgger 정답).
        *   **Defaults**: Epochs=20, Batch_Size=32, WORKER_FREQ=2 (Worker 10회 학습).
    2.  **Phase 2: Reinforcement Fine-tuning (`train_rl.py`)**:
        *   POMO 알고리즘을 통한 실전 미세 조정.
        *   **Defaults**: Episodes=5000, Batch_Size=24 (Low VRAM Mode), Hidden_Dim=256.
    3.  **Phase 3: Visualization & Evaluation (`tests/evaluate.py`)**:
        *   분산되어 있던 시각화, 평가 파이프라인을 3-파일 엔진 구조(`eval_core.py`, `eval_viz.py`, `evaluate.py`)로 최적화했습니다.
        *   `eval_core.py`: 모델 로드, 롤아웃 수행, A* 최단 경로 탐색 등 코어 엔진 역할 수행.
        *   `eval_viz.py`: Matplotlib 등 모든 시각화 렌더링 로직 독립 관리.
        *   `evaluate.py`: 여러 실험 환경(dashboard, paper_full, regen 등 9개 서브커맨드)을 일괄 관장하는 단일 진입점(CLI) 역할.

### 2.2 학습 루프 상세 (Training Loop Details)
#### **RL Loop (`train_rl.py`, 2026-04-06 기준)**

현재 [train_rl.py](/d:/연구실/연구/재난드론/Code/train_rl.py)는 `train_rl()` 단일 함수로 구성됩니다. (2026-04-06 리팩토링으로 `_run_stage()` 제거, 단일 진입점화)

추가된 인자:
*   `--wkr_lr_floor` (기본 `1e-5`): Worker 최소 학습률
*   Critic shape mismatch 자동 감지 및 `strict=False` 로드 안전장치

##### A. `train_rl(args)`
상위 orchestration 함수이며, 두 가지 모드를 지원합니다.

1.  **단일 stage 실행**
    *   `manager`, `worker`, `joint`, `--disaster`
2.  **Phase 1 전체 파이프라인 실행**
    *   `--stage phase1`
    *   내부적으로 `manager -> worker -> readiness check -> joint(optional)`를 순차 실행

추가로, `--stage joint`와 `--stage phase1` 모두 `--force_joint`를 지원합니다.

*   readiness fail 상태에서는 기본적으로 joint 실행을 막음
*   `--force_joint=True`이면 research/smoke 목적으로 gate를 우회 가능

##### B. `_run_stage(args, stage_override=None)`
실제 한 stage를 실행하는 low-level 함수입니다. 내부 흐름은 다음과 같습니다.

1.  **Environment 생성**
    *   `DisasterEnv(..., enable_disaster=args.disaster)`
2.  **Model 생성**
    *   `GraphTransformerManager(node_dim=4, hidden_dim=args.hidden_dim, dropout=0.2)`
    *   `WorkerLSTM(node_dim=8, hidden_dim=args.hidden_dim)` (Critic: 2-Layer MLP)
3.  **Checkpoint 로드**
    *   `manager` / `worker` stage: `logs/sl_pretrain/model_sl_final.pt`
    *   `joint` stage:
        *   manager는 `logs/rl_phase1_manager/best.pt`
        *   worker는 `logs/rl_phase1_worker/best.pt`
        *   없으면 SL checkpoint fallback
    *   `worker` stage:
        *   manager는 가능하면 `logs/rl_phase1_manager/best.pt`
        *   없으면 `logs/sl_pretrain/model_sl_final.pt` fallback
        *   worker는 `logs/sl_pretrain/model_sl_final.pt`
    *   `--disaster`:
        *   `logs/rl_phase1_joint/best.pt`
4.  **Config 구성**
    *   `lr`
    *   `num_pomo`
    *   `save_dir`
    *   `stage`
    *   `debug`
    *   `mgr_lr_scale=0.5`
    *   `mgr_eta_min_scale=0.1`
    *   `mgr_max_grad_norm=10.0`
    *   `wkr_max_grad_norm=5.0`
    *   `mgr_aux_start/end = 0.20 / 0.05`
    *   `wkr_aux_start/end = 0.20 / 0.05`
5.  **Trainer Dispatch**
    *   `ManagerStageTrainer`
    *   `WorkerNavTrainer`
    *   `DOMOTrainer`
6.  **`trainer.train(args.episodes)` 실행**

##### C. `phase1` readiness와 산출물
`--stage phase1`는 manager와 worker stage가 끝난 뒤, 최신 debug row를 읽어 readiness를 계산합니다.

*   산출물:
    *   `logs/rl_phase1_joint/stage_readiness.json`
    *   `logs/rl_phase1_joint/val_by_refbin.csv`
*   현재 readiness는 두 단계입니다.
    *   `smoke_ready`
    *   `launch_ready`

현재 구현된 launch gate 기준은 다음과 같습니다.

*   `manager.plan_under_rate < 0.25`
*   `manager.anchor_near_rate > 0.20`
*   `manager.first_subgoal_hops_mean <= 6.0`
*   `worker.stagnation_fail_rate < 0.30`
*   `worker.goal_after_last_subgoal_rate > 0.70`
*   `worker.post_last_sg_success_rate > 0.60`

즉, 현재 `phase1`은 “무조건 joint까지 간다”가 아니라, **manager/worker가 최소 기준을 통과했을 때만 joint를 자동 실행**하는 구조입니다.

##### D. 현재 trainer별 역할 분담
*   **`ManagerStageTrainer`**
    *   Worker freeze
    *   sparse warm-start + plan-only score + auxiliary CE로 Manager 학습
*   **`WorkerNavTrainer`**
    *   Manager freeze
    *   A* 기반 Hidden Checkpoint를 따라 Worker를 네비게이션 시키는 독립적 집중 훈련
*   **`DOMOTrainer`**
    *   joint stage와 disaster stage의 공통 trainer
    *   실제 rollout reward + manager/worker auxiliary loss를 함께 사용

##### E. 저장 산출물
모든 stage는 공통적으로 다음 파일들을 남깁니다.

*   `best.pt`
*   `final.pt`
*   `rl_learning_curve.png`
*   `rl_debug_log.txt`
*   `debug_metrics.csv`
*   `debug_episode_sample.jsonl`

즉, 현재 문서에서 RL spec은 “단일 trainer로 모든 것을 다 처리한다”가 아니라, **entrypoint는 하나지만 trainer는 stage별로 분리되어 있다**고 이해하는 것이 정확합니다.

### 2.2 물리 시뮬레이션 흐름 (Physics Simulation Flow)
`env.step()` 내부에서 일어나는 일입니다.
1.  **입력**: 현재 노드($U$), 이동할 노드($V$)
2.  **데이터 조회**: $U \to V$ 엣지의 상태(Normal/Caution/Closed)와 파괴도(0.0~1.0)를 확인.
3.  **물리 연산 (`robot.py`)**:
    *   **Simplified**: 에너지/디테일 파괴도 연산 제거.
    *   `속도 = 기본속도(40km/h)`
    *   `에너지 = N/A`
4.  **상태 갱신**:
    *   `Current_Node` 업데이트.
    *   `Current_SOC` 업데이트 (미사용).

---

## 3. 환경 계층 (Environment Layer)

### 2.1 `src/envs/disaster_env.py`

#### **Class `DisasterEnv`**
*   **Tensor Shapes**:
    *   `pos_tensor`: `[Num_Nodes, 2]` (Coordinates)
    *   `x`: `[Batch * N, 8]` (Node Features: `[pos_x, pos_y, is_cur, is_tgt, visit, dist, dir_x, dir_y]`)
    *   `edge_index`: `[2, Num_Physical_Edges]` (Topology)
    *   `damage_states`: `[Batch_Size, Num_Physical_Edges]` (0: Normal, 1: Damaged) [Binary State]
    *   `seismic_schedule`: Dict `[Step]` -> `Params` (Unified Shock Model)

#### **Method `reset(batch_size)`**
1.  **Unified Shock Model (지진 스케줄링)**:
    *   에피소드당 $3 \sim 5$회의 지진(`Shock`)을 스케줄링.
    *   **강도 혼합**: 강진($6.0 \sim 7.5$)과 중/약진($4.5 \sim 6.5$)을 무작위 순서(Shuffle)로 배치.
    *   `self.seismic_schedule`에 이벤트 타입, 강도 범위, 진앙지 수 저장.
    *   $t=0$ 시점에 첫 번째 지진(`Shock_0`) 즉시 트리거.
2.  **Physics Init**:
    *   `_apply_fragility`: 초기 파괴 상태 계산 (Threshold + Noise).
    *   `_update_edge_attributes`: 파괴된 엣지의 `Status`를 'Blocked'로 설정 (Speed=0).

#### **Method `step(action)`**
1.  **Seismic Trigger**:
    *   현재 `step_count`가 스케줄에 있다면 지진(`Shock_t`) 발생.
    *   **Epicenter Shift**: 여진의 진앙지는 이전 진앙지의 '이웃 노드'로 무작위 이동 (동적 위험 확산).
    *   **Progressive Damage**: `damage_states`는 누적됨 (Monotonic: $0 \to 1$).
2.  **Reward & Done**: 도착 여부 및 배터리 소진 체크.

#### **Internal Physics Logic (`_apply_fragility`)**
*   **역활**: HAZUS 강도(Intensity)와 확률적 불확실성을 결합하여 파괴 여부 결정.
*   **공식**: $Damaged \iff Intensity > (Threshold + Noise)$
*   **Threshold (HAZUS Linked)**:
    *   **Bridge**: $3.8g$ (Moderate/Extensive 경계 위험)
    *   **Road**: $5.0g$ (Complete 진입 시 위험)
*   **Uncertainty**: Noise $\epsilon \sim N(0, 0.4)$
    *   HAZUS 등급과 실제 파괴 상태 간의 불일치를 유발하여 RL 에이전트의 불확실성 적응 유도.
    *   **Dynamic Edge Attributes Update [Critical Fix]**:
        *   기존 Static `edge_attr` 대신, 매 에피소드마다 변하는 재난 상황을 `pyg_data.edge_attr`에 주입.
        *   **Col 1**: Damage (0.0 ~ 1.0)
        *   **Col 2**: Expected Time (Closed=100.0)
        *   **Col 3**: Expected Energy (Closed=100.0)
        *   **Col 4**: Is_Closed (1.0 or 0.0)
    *   **Distance Curriculum (Range-Based Sliding Window)**:
        *   **목적**: 난이도를 정밀하게 제어하기 위해 최소 거리만 제한하는 것이 아니라 **특정 거리 구간(Window)**의 문제만 출제.
        *   **Window Size**: 전체 맵 최대 거리의 **20%**.
        *   **Logic**:
            *   `Min_Dist = Max_Dist * 0.8 * Ratio` (0% ~ 80% 이동)
            *   `Max_Dist = Min_Dist + (Max_Dist * 0.2)` (Min + 20% 구간)
        *   **Effect**: 학습 초기에는 확실히 쉬운(가까운) 문제만, 후반에는 확실히 어려운(먼) 문제만 집중적으로 학습.
    *   **Visualization Data Storage**:
        *   `self.pga`: `[Batch, NumEdges]` (Peak Ground Acceleration).
        *   **Purpose**: HAZUS 5단계 시각화(Normal ~ Complete)를 위해 물리 엔진 내부의 PGA 값을 보존합니다.
    *  ### 3.1. Stochastic Network Diffusion Model (확률적 네트워크 확산 모델)
기존의 물리학(GMPE) 기반 모델을 대체하여, 그래프 이론과 확률론적 접근을 통해 피해를 시뮬레이션합니다.

*   **핵심 로직 (Core Logic)**:
    1.  **Seed Selection**: 전체 노드 중 $k$개의 'Epicenter Node'를 무작위 선택.
    2.  **Graph Diffusion**: APSP(All-Pairs Shortest Path) 행렬을 사용하여 진앙지로부터의 **네트워크 거리(Network Distance)** 계산.
    3.  **Intensity Score**: $Score = Base - (Decay \times Dist_{net}) + Noise$
        *   $Base \sim U(7.0, 8.5)$
        *   $Decay = 2.0 \times \log_{10}(Dist_{km} + 1.0)$
        *   $Noise \sim N(0, 1)$ (Clamped $[-2.0, 2.0]$)

### 3.2. Probabilistic Fragility (확률적 파괴 모델)
점수(Score)에 따라 구조물별로 다른 확률로 파괴 상태가 결정됩니다.

*   **교량 (Bridges)**: 취약 설비
    *   **Collapse ($S=2$)**: $Score > 5.5$ 일 때 **80% 확률**. ($Score > 4.0$ 일 때 30%)
    *   **Damage ($S=1$)**: $Score > 4.0$ 일 때 **90% 확률**.
    *   **Degradation**: 이미 손상($S=1$)된 경우, 추가 붕괴 확률 **+50%p**.

*   **일반 도로 (Roads)**: 내구 설비
    *   **Collapse ($S=2$)**: $Score > 7.0$ (매우 높은 강도) 일 때만 **10% 확률**.
    *   **Damage ($S=1$)**: $Score > 5.0$ 일 때 60% 확률.
    *   **Degradation**: 손상 시 추가 붕괴 확률 **+10%p** (잘 무너지지 않음).
    *   **Physics Parameters (HAZUS Based)**:
        *   **GMPE (Ground Motion Prediction Equation)**:
            *   $\ln(PGA) = C_1 + C_2 \cdot M - C_3 \cdot \ln(R + C_4)$
            *   $C_1=-0.5, C_2=0.9, C_3=1.1, C_4=10.0, \sigma=0.6$
            *   **Magnitude**: $5.0 \sim 6.0 M_w$ (Reduced for stability)
            *   **Epicenter**: Safe Generation (Rejection Sampling, Min Dist > 50% Map Width)
            *   **Coordinate Scale**: Lat/Lon 좌표(예: Sioux Falls)일 경우 Haversine 근사(1deg Lat $\approx$ 111km)를 통해 $R$을 km 단위로 자동 변환.
        *   **Fragility (Tunnel/Bridge)**:
            *   $\beta=0.6$ (Standard Dispersion)
            *   $\theta_{normal}=0.80g$ (Complete Damage Threshold)
            *   **Diversity Multipliers**:
            *   **Diversity Modifiers (Additive - User Request)**:
                *   **Highways (High Speed & Connected)**: $\theta += 0.15g$
                *   **Logic**: `Speed >= 4842`인 도로 중, **5개 이상의 엣지로 연결된 컴포넌트**만 필터링 (Ramp 포함).
                *   **Others**: $\theta -= 0.05g$
                *   **(Note)**: Replaced multiplicative logic to prevent excessive range shift.
                *   **Noise**: $\theta += N(0, 0.05)$
            *   $\theta_{damaged}=0.25g$ (Weakened Threshold)
        *   **Edge Attributes**:
            *   **Speed Factor**: Normal(1.0) / Damaged(0.5) / Collapsed(0.0)
            *   **Cost Ratio**: Normal(0.0) / Damaged(0.2) / Collapsed(1.0)
    *   **Battery Budget Calculation (Relaxed)**:
        *   **도로망 최단거리(APSP)** 기준 에너지 산출.
        *   `Allocated_Energy = Network_Dist * Rated_Eff(Normal) * 1.5`
        *   **제약 완화**: 기존 1.2배에서 **1.5배(50% 여유분)**로 늘려, 에이전트가 우회로를 위해 에너지를 사용할 수 있는 여지를 확보. 단, 물리적 최대 용량(36MJ)를 초과할 수 없음.

#### **Method `update_target_features(new_targets)` [NEW]**
*   **Role**: Subgoal 변경 시 관련된 Worker 입력 피처를 일괄 업데이트.
*   **Updates**:
    *   `is_target` (Col 3): 새로운 타겟 위치에 1.0 설정.
    *   `net_dist` (Col 5): APSP 행렬을 조회하여 정규화된 거리 업데이트.
    *   `direction` (Col 6, 7): 현재 노드에서 타겟을 향한 단위 벡터 계산.

#### **Method `step(action)`**
```python
def step(self, next_node_idx: torch.Tensor) -> None:
```
*   **Data Flow**:
    1.  `curr = self.current_node`, `next = next_node_idx`
    2.  **Energy Consumption**:
        *   `current_soc` 업데이트.
    3.  **State Sync**:
        *   `pyg_data.x`의 `is_current` (Col 2) 및 `visit_count` (Col 4)를 In-place 업데이트하여 GNN 입력 동기화.
    4.  **Aftershock Trigger**: (Probabilistic)
        *   여진 발생 시 진원지 이동 및 규모 감소.
        *   `_generate_ground_motion` -> `_apply_fragility` 재실행.
        *   `damage_states` 및 엣지 속성 업데이트.

#### **RL 보상 함수 (2026-03-31 Stage-Aware Design)**
현재 RL 보상은 더 이상 단일 형태가 아닙니다. [pomo_trainer.py](/d:/연구실/연구/재난드론/Code/src/trainers/pomo_trainer.py), [manager_stage_trainer.py](/d:/연구실/연구/재난드론/Code/src/trainers/manager_stage_trainer.py), [worker_stage_trainer.py](/d:/연구실/연구/재난드론/Code/src/trainers/worker_stage_trainer.py)에서 **stage별로 서로 다른 목적 함수**를 사용합니다.

핵심 철학은 다음과 같습니다.

*   **Manager stage**:
    *   실제 goal success보다 **plan 품질**을 먼저 학습
*   **Worker stage**:
    *   plan 품질 penalty를 제거하고 **reference plan 추종 능력**만 학습
*   **Joint stage**:
    *   rollout reward + 약한 imitation regularization으로 **end-to-end alignment** 수행

### A. 공통 rollout reward (`DOMOTrainer.execute_batch_plan`)
joint/disaster/worker rollout에서 공통적으로 쓰는 기본 보상은 아래 요소들로 구성됩니다.

*   `R1 PBRS`
*   `R2 Subgoal`
*   `R3 Goal`
*   `R4 Efficiency`
*   `R5 Milestone`
*   `R6 Explore`
*   `P1 Time`
*   `P2 Loop`
*   `P3 Fail`
*   `Plan Adjustment` (stage에 따라 다름)

현재 [pomo_trainer.py](/d:/연구실/연구/재난드론/Code/src/trainers/pomo_trainer.py)의 공통 기본값은 다음과 같습니다.

*   `POTENTIAL_SCALE = 6.5`
*   `SUBGOAL_BASE = 0.5`
*   `SUBGOAL_SCALE = 1.5`
*   `OPTIMALITY_BONUS = 0.5`
*   `GOAL_REWARD = 40.0`
*   `EFFICIENCY_MAX = 8.0`
*   `MILESTONE_25/50/75 = 0.75 / 1.5 / 3.0`
*   `EXPLORATION_BONUS = 0.0`
*   `BASE_STEP_PENALTY = -0.02`
*   `TIME_PRESSURE_SCALE = 2.0`
*   `LOOP_PENALTY_SCALE = 0.3`
*   `FAIL_PENALTY = -20.0`

중요한 점은 현재 `R1 PBRS`가 더 이상 모든 stage에서 같은 target을 바라보지 않는다는 것입니다.

#### Stage-aware PBRS
현재 rollout은 `active target` 개념을 명시적으로 사용합니다.

*   active subgoal이 남아 있으면 `active target = current subgoal`
*   더 이상 유효 subgoal이 없으면 `active target = goal`
*   goal은 virtual final target처럼 취급되어, 마지막 subgoal 이후 handoff가 명시적으로 추적됩니다

stage별 PBRS 사용 방식은 다음과 같습니다.

*   `worker-only`
    *   `active target progress`만 shaping에 사용
    *   `lambda_sg = 1.0`, `lambda_goal = 0.0`
*   `joint/disaster` pre-handoff
    *   `active target progress + 작은 final-goal tether`
    *   `lambda_sg = 1.0`, `lambda_goal = 0.2`
*   `post-last-subgoal`
    *   `goal-only shaping`

또한 pointer가 바뀌는 step에서는 potential을 새 target 기준으로 rebase하여, target switch 자체가 거대한 reward jump를 만들지 않도록 처리합니다.

#### Subgoal Bonus 축소
기존 explicit subgoal bonus는 중복 보상을 줄이기 위해 50% 축소되었습니다.

*   현재 구현:
    *   `0.5 * (SUBGOAL_BASE + SUBGOAL_SCALE * progress_ratio)`
*   그리고 “첫 도달 시 1회만 지급” 구조를 유지합니다.

### B. Handoff / Arrival / Stagnation 규칙
현재 Worker와 Joint rollout에서 가장 중요한 안정화 패치는 `soft-arrival`, `adaptive skip`, `adaptive stagnation`입니다.

#### 도착 판정 (2026-04-06 완화 강화)
현재 subgoal 도착은 exact-arrival만 사용하지 않습니다.

*   `curr == sg`면 exact 도달
*   `hop(curr, sg) <= 1`이면 near 상태
*   near 상태이면서
    *   마지막 target이거나
    *   `hop(curr, next_target) + 1 <= hop(sg, next_target)`이거나
    *   **`approaching_next`: `hop(curr, next_target) < hop(prev, next_target)` (다음 목표에 접근 중)**이면
    *   `soft-arrival` 허용

즉, subgoal을 사실상 지나갔는데도 exact node를 못 밟아서 멈추는 현상을 줄이는 방향입니다.

#### Adaptive Skip
현재 target이 막혔거나 사실상 도달 불능이며, 다음 target이나 goal 쪽으로 실제 개선이 있는 경우 `skip`을 허용합니다.

#### Adaptive Stagnation
stagnation patience는 고정값이 아니라 아래 식으로 계산됩니다.

*   `clamp(10 + 2 * init_active_hops, 12, 40)`

그리고 아래 중 하나가 발생하면 stagnation 카운터를 reset합니다.

*   active target hop 개선
*   goal distance 감소
*   pointer 변경
*   goal 도달
*   soft-arrival
*   skip

마지막 subgoal 이후 goal handoff가 발생하면 `stagnation_steps`, `best_subgoal_hops`, `segment_start_goal_dist`, `steps_since_last_subgoal`도 함께 초기화합니다.

### C. Manager Plan Geometry 관련 공통 개념
Phase 1의 Manager 관련 stage는 아래 개념을 공유합니다.

*   `shortest_hops = hop(start, goal)`
*   `TARGET_SEGMENT_HOPS = 4.5`
*   `plan_len_ref = ceil(shortest_hops / 4.5)`
*   `plan_len_min = 1 if shortest_hops <= 4 else plan_len_ref`
*   `plan_len_max = plan_len_ref + 1`
*   reference anchor는 **hop shortest path**를 균등 간격으로 샘플링해 생성

즉, 현재 Phase 1에서는 weighted shortest path가 아니라 **hop 기반 sparse plan geometry**를 기준 좌표계로 사용합니다.

#### Budget-Aware Decode
Manager decode는 이제 아래 budget-aware radius를 사용합니다.

*   `remaining_hops = hop(current, goal)`
*   `remaining_slots = max(1, k_ref - generated_so_far)`
*   `seg_ref = remaining_hops / remaining_slots`
*   `r_min = max(2, floor(0.5 * seg_ref))`
*   `r_max = ceil(1.5 * seg_ref) + 1`

decode에서:

*   `hop > r_max` 후보는 hard mask
*   `hop < r_min` 후보는 soft penalty
*   기존 135도 cone hard mask는 제거하고 soft directional bias로 교체

즉, Manager는 현재 “임의의 3~10 hop 후보”가 아니라 “남은 segment budget에 맞는 hop 길이”를 기준으로 sparse plan을 생성합니다.

### D. Manager-only Stage의 Plan Score (2026-04-18 안정화 업데이트)
Manager-only stage는 실제 worker rollout reward 대신, **plan 자체의 구조적 품질 점수**를 직접 최적화합니다.

현재 [manager_stage_trainer.py](file:///home/sem/Juhyeong/Code/src/trainers/manager_stage_trainer.py)의 점수 구성은 다음과 같습니다.

*   `R_count` (2026-04-18 업데이트)
    *   `-1.20 * under_count - 0.15 * over_count + 0.50 * in_range`
    *   **변경:** 적정 범위(plan_len_min~plan_len_max) 생성 시 **+0.50 보너스** 추가 (보상 해킹 방지)
*   `R_anchor`
    *   `-0.40 * anchor_hop_error_mean + 0.95 * anchor_near_ratio`
*   `R_spacing`
    *   `-1.20 * spacing_error_mean`
*   `R_mono`
    *   `-0.80 * monotonic_violation_mean`
*   `R_budget`
    *   `-1.10 * segment_budget_error_mean`
*   `R_front`
    *   `-1.50 * first_segment_overshoot`
*   `R_first`
    *   첫 subgoal이 기대 첫 anchor보다 너무 먼 경우와 첫 segment budget error에 대한 penalty
*   `R_corr`
    *   corridor deficit이 클 때만 약하게 penalty (`-0.20 * deficit`)
*   `R_empty` (2026-04-18 강화)
    *   빈 계획이면 **`-5.0`** (이전 `-2.0`에서 상향)
    *   **변경 근거:** EOS 즉시 출력 전략이 너무 저렴하여 Manager가 "겁쟁이 편향(Coward Bias)"을 학습하던 문제 방지

**Advantage Normalization:**
```python
normalized_plan_score = (plan_score - plan_score.mean()) / (plan_score.std(unbiased=False) + 1e-8)
```
Mean 0, Std 1 정규화가 Policy Loss 역전파 전에 적용되어 gradient 스케일 안정화.

#### 학습 안정화 하이퍼파라미터 (2026-04-18)

| 파라미터 | 이전값 | 현재값 | 변경 근거 |
|---------|--------|--------|----------|
| `mgr_max_grad_norm` | 20.0 | **5.0** | Pre-clip norm 100~800 폭등으로 SL 가중치 파괴 |
| `sparse_warmstart_ratio` | 0.20 | **0.40** | 초반 RL 노이즈로부터 SL 지식 보호 |
| LR `hold_ratio` | 0.7 | **0.4** | 높은 LR 장기 유지가 gradient 폭발 가속 |
| LR `min_factor` | 0.3 | **0.1** | 후반 LR을 더 낮춰 미세 조정 |
| RL 최대 비중 | 0.70 | **0.50** | RL 비중 0.70에서 Plan Score 역행 관측 |
| Empty plan 페널티 | -2.0 | **-5.0** | 겁쟁이 전략 비용 증가 |
| In-range 보너스 | 0.0 | **+0.50** | 적정 길이 플랜 생성 유인 |

#### Sparse Warm-start
Manager stage 초반 **`40%`** episode는 sparse anchor sequence CE 중심 warm-start입니다.

*   `RL weight = 0.0`
*   `Aux weight = 1.0`

이후 나머지 구간에서 보수적 비율의 RL + auxiliary CE 혼합으로 전환합니다(최대 RL 0.50).

### E. Worker-only Stage의 Loss와 Curriculum
Worker-only stage는 manager plan penalty를 직접 쓰지 않습니다. 대신 다음 loss를 사용합니다.

*   `policy_loss`
*   `0.5 * critic_loss`
*   `worker_aux_ce_weight * worker_aux_ce_loss`
*   `-0.01 * worker_entropy`

여기서 `worker_aux_ce_loss`는:

*   target subgoal까지 가는 optimal next-hop을 `weighted_next_hop_matrix[current_node, target]`에서 조회
*   현재 node가 path에서 벗어나도, 그 node 기준 expert next-hop을 다시 조회하는 **DAgger 스타일 supervision**

또한 Worker-only stage는 이제 clean reference sparse plan만 보지 않습니다.

*   초반 `70%`
    *   `80% clean reference + 20% perturbed anchor`
*   후반 `30%`
    *   `50% clean reference + 30% perturbed anchor + 20% manager-sampled plan`

그리고 마지막 subgoal 이후 goal이 active target이 된 직후 `8 step` 동안, worker auxiliary CE에 `handoff_aux_boost = 1.5`가 적용됩니다.

Worker-only stage는 `detach_spatial=False`이므로 spatial encoder까지 실제로 적응합니다.

### F. Joint Stage의 Plan Adjustment
Joint stage는 Manager stage보다 훨씬 약한 plan shaping만 사용합니다. 현재 [pomo_trainer.py](/d:/연구실/연구/재난드론/Code/src/trainers/pomo_trainer.py)의 joint 기본값은 다음과 같습니다.

*   `PLAN_CORRIDOR_WEIGHT = 1.5`
*   `PLAN_COUNT_UNDER_PENALTY = 1.0`
*   `PLAN_COUNT_OVER_PENALTY = 0.2`
*   `ANCHOR_HOP_PENALTY = 0.15`
*   `ANCHOR_NEAR_BONUS = 0.50`
*   `FIRST_ANCHOR_PENALTY = 0.30`
*   `SPACING_PENALTY_SCALE = 0.80`
*   `MONOTONIC_PENALTY_SCALE = 0.50`
*   `PLAN_ADJUST_MIN = -6.0`
*   `PLAN_ADJUST_MAX = 4.0`

의미:

*   Manager-only stage보다 압박이 약함
*   이미 분리 stage에서 학습한 plan prior를 joint interaction 속에서 살짝 정렬하는 수준

#### Joint Readiness
Joint는 현재 기본적으로 readiness gate를 통과해야만 자동 실행됩니다.

*   `smoke_ready`
    *   `plan_under_rate < 0.40`
    *   `anchor_near_rate > 0.10`
    *   `stagnation_fail_rate < 0.50`
    *   `goal_after_last_subgoal_rate > 0.50`
*   `launch_ready`
    *   `plan_under_rate < 0.25`
    *   `anchor_near_rate > 0.20`
    *   `first_subgoal_hops_mean <= 6.0`
    *   `stagnation_fail_rate < 0.30`
    *   `goal_after_last_subgoal_rate > 0.70`
    *   `post_last_sg_success_rate > 0.60`

즉, joint는 현재 “무조건 Phase 1의 마지막 단계”가 아니라, **Manager/Worker 인터페이스가 최소 기준을 만족했을 때만 자동으로 열리는 stage**입니다.

### G. Auxiliary Imitation Loss
현재 RL은 pure policy gradient만 쓰지 않습니다.

#### Manager Auxiliary CE
*   reference anchor sequence를 target으로 삼아 CE loss 계산
*   joint stage 기준 weight:
    *   `mgr_aux_start = 0.20`
    *   `mgr_aux_end = 0.05`

#### Worker Auxiliary CE
*   expert next-hop을 target으로 CE loss 계산
*   joint stage 기준 weight:
    *   `wkr_aux_start = 0.20`
    *   `wkr_aux_end = 0.05`

이 보조 loss의 목적은 RL 중에도 SL prior가 완전히 무너지지 않게 하는 것입니다.

### H. 현재 보상 해석 기준
2026-03-31 기준, 이제 corridor는 1순위 지표가 아닙니다. 아래 지표를 우선적으로 해석해야 합니다.

*   `plan_under_rate`
*   `plan_len_mean / plan_len_ref_mean`
*   `anchor_hop_err_mean`
*   `anchor_near_rate`
*   `first_subgoal_hops_mean`
*   `segment_budget_error_mean`
*   `first_segment_budget_err_mean`
*   `frontloaded_overshoot_rate`
*   `goal_after_last_subgoal_rate`
*   `post_last_sg_success_rate`
*   `soft_arrival_rate`
*   `skip_rate`
*   `stagnation_fail_rate`
*   `manager_clip_hit_rate`
*   `success_ema`

이유는 현재 failure mode가 “corridor 밖으로 샌다”보다

*   Manager 쪽은 **길이와 배치, 특히 first-segment budget**
*   Worker 쪽은 **last SG -> goal handoff와 stagnation**

문제에 더 가깝기 때문입니다.

#### **Critic / Advantage 안정화 (2026-03)**
기존 raw return 기반 critic은 reward scale 변화와 outlier에 매우 취약했습니다. 이를 완화하기 위해 critic은 이제 **normalized return**을 예측합니다.

*   **정규화 방식**:
    *   trainer가 `ret_ema_mean`, `ret_ema_std`를 EMA(`momentum=0.99`)로 유지
    *   `norm_return = (return - ret_ema_mean) / (ret_ema_std + 1e-6)`
*   **Critic Target**:
    *   Worker critic은 raw return이 아니라 `norm_return`을 회귀
*   **Actor Baseline**:
    *   Advantage 계산도 같은 normalized scale에서 수행
    *   `advantage = norm_return - V0`
    *   이후 batch-wise 표준화 적용
*   **Critic Loss**:
    *   `MSE` 대신 `SmoothL1Loss(beta=1.0)` 사용
*   **의도**:
    *   outlier trajectory가 critic을 망가뜨리는 현상 완화
    *   reward 재설계 후에도 baseline을 안정적으로 유지

#### **RL 디버그 로그 해석 가이드 (`--debug`)**
`train_rl.py --debug`는 단일 episode가 아니라 최근 window(보통 200 episode)의 평균 통계를 출력합니다. 각 섹션은 아래와 같이 해석합니다.

*   **1. Reward Alignment**
    *   `Goal Hit Rate`: `R3 Goal / GOAL_REWARD`로 계산한 실제 goal 도달률.
    *   `Success/Fail`: 성공 trajectory 평균 final reward / 실패 trajectory 평균 final reward.
    *   `Norm Ret S/F`: 성공/실패 trajectory의 normalized return 평균.
    *   `Fail Shaping`: 실패 trajectory에서 `R1 + R2 + R5 + R6`의 평균.
    *   `Goal Share`: 성공 trajectory final reward 중 `R3 Goal`의 비중.
    *   **좋은 상태**:
        *   `Fail Shaping < 10`
        *   `Goal Share > 0.5`
        *   성공/실패 보상이 명확히 분리

*   **2. Batch Outcome**
    *   `Success Rate`: 실제 goal 도달 비율.
    *   `Success EMA`: success rate의 이동 평균.
    *   `Loop Fail`: 루프/비정상 종료 비율.
    *   `Goal Dist`: episode 종료 시 goal까지 남은 APSP 거리.
    *   `Progress`: 시작점 대비 goal까지 얼마나 가까워졌는지의 진행률.
    *   **좋은 상태**:
        *   `Success Rate`와 `Success EMA`가 꾸준히 상승
        *   `Goal Dist` 감소
        *   `Progress` 평균이 양수

*   **3. Plan Diagnostics**
    *   `Plan Length`: Manager가 생성한 유효 subgoal 수.
    *   `EOS Rate`: plan에 EOS가 포함된 비율.
    *   `Empty Plans`: subgoal 없이 끝난 비율.
    *   `Unique Ratio`: plan 내 중복되지 않는 node 비율.
    *   `Density`: `generated_subgoals / shortest_path_hops`
    *   `band`: expert waypoint spacing 기준 이상적 density 범위 (`1/11 ~ 1/5`)
    *   `gap`: density가 이상 범위에서 얼마나 벗어났는지
    *   `Corridor`: 각 subgoal이 `start->sg + sg->goal <= shortest + 2` 조건을 만족하는 비율
    *   `SG Hops`: 첫 subgoal까지 hop 수 / subgoal 간 평균 hop / 최대 hop
    *   **좋은 상태**:
        *   `Empty Plans = 0%`
        *   `Density < 1.0` 또는 강한 하락 추세
        *   `Corridor > 0.6`
        *   `Unique Ratio` 높음

*   **4. Worker Diagnostics**
    *   `Subgoal Reach`, `Plan Utiliz.`:
        *   실제로 밟은 subgoal 비율.
        *   현재 acceptance target은 `> 0.25`.
    *   `Hit@1/2/3/4/5+`:
        *   k번째 rank의 subgoal까지 도달한 비율.
        *   초반 rank만 높고 뒤가 급락하면 plan이 너무 길거나 Worker가 중후반 plan을 못 따라가는 상태입니다.
    *   `Active SG Hop`:
        *   active subgoal에 가장 근접했을 때의 hop 거리.
        *   0에 가까울수록 좋습니다.
    *   `Max SG Prog.`:
        *   active subgoal 기준 최대 진행률.
    *   `Goal/SG per100`:
        *   100 step당 goal progress / subgoal progress.
        *   goal 값이 음수면 subgoal은 따라가더라도 실제 목적지로는 멀어지고 있다는 뜻입니다.

*   **5. Critic & Advantage**
    *   `Return Mean`: raw final reward 평균
    *   `Norm Return`: critic target으로 쓰는 normalized return 평균/표준편차
    *   `Value Mean`: critic이 예측한 normalized value 평균
    *   `TD |err|`: critic 예측 오차 크기
    *   `Expl. Var`: critic 설명력
        *   `1.0`: 매우 좋음
        *   `0.0`: 평균 예측 수준
        *   `< 0`: 평균보다 못함
    *   **좋은 상태**:
        *   `Norm Return std ≈ 1`
        *   `Expl. Var > 0.1`
        *   `TD |err|` 감소 추세

*   **6. Gradient Norms**
    *   `pre`: clipping 전 gradient norm
    *   `post`: clipping 후 gradient norm
    *   `clip-hit`: clipping 상한에 걸린 비율
    *   **좋은 상태**:
        *   `manager_clip_hit_rate < 90%`
        *   `worker_clip_hit_rate < 90%`
    *   **나쁜 신호**:
        *   둘 다 계속 `100%`면 학습이 항상 잘린 gradient로만 진행되는 상태

*   **7. Manager Diagnostics**
    *   `Mgr Entropy`: Manager token 분포의 엔트로피
    *   `Mgr NLL`: Manager가 자기 sequence를 얼마나 높은 확률로 냈는지 보는 지표
    *   `Policy Loss`, `Critic Loss`, `Ent Bonus`: RL 최적화 내부 항목
    *   **해석 팁**:
        *   부호보다 추세를 보아야 합니다.
        *   success/plan metrics가 개선되지 않는데 loss만 출렁이면 objective mismatch 가능성이 큽니다.

#### **현재 실패 패턴의 전형적 해석**
최근 디버그 로그에서 자주 보이는 실패 패턴은 아래와 같습니다.

*   `R6 Explore = 0`, `Fail Shaping`은 낮아졌지만 `Success Rate`가 여전히 낮다:
    *   reward alignment는 개선되었으나, plan 실행 가능성(plan executability) 문제가 남아 있음.
*   `R7 PlanPen.`이 큰 음수이고 `Density`가 높다:
    *   Manager가 여전히 너무 촘촘한 plan을 생성.
*   `Corridor`가 낮고 `Plan Utiliz.`가 낮다:
    *   plan이 구조적으로도 좋지 않고, Worker도 그 계획을 충분히 사용하지 못함.
*   `Goal/SG per100`에서 goal은 음수, subgoal은 양수:
    *   subgoal은 일부 따라가지만 최종 goal과 정렬되지 않음.
*   `Expl. Var <= 0` 또는 `clip-hit = 100%`:
    *   critic/gradient 안정성이 아직 부족하며 추가 안정화가 필요함.

---

### 2.3 물리 시뮬레이션 상세 (Advanced Physics Engine)
`src/envs/disaster_env.py` 내부에 구현된 고도화된 재난 물리 엔진입니다.

#### **1. 공간 상관관계 (Spatial Correlation)**
*   **구현**: `_init_physics_engine()`
*   **원리**: 모든 도로 엣지의 중심점 간 거리($d$)를 기반으로 상관 행렬 구축.
*   **공식**: $\rho(d) = \exp(-d / 2.0)$
*   **최적화**: Cholesky 분해($L$)를 미리 계산하여 저장 (`[NumEdges, NumEdges]`).

#### **2. 지진동 생성 (Ground Motion Generation)**
*   **Method**: `_generate_ground_motion`
*   **Seismic Sequence (Updated 2026-02-04)**: 
    *   **Schedule**: `reset()` 시점에 전체 에피소드의 지진 발생 스케줄 (`self.seismic_schedule`) 생성.
    *   **Pattern**: Main Shock (Strong) -> Random Delay -> Aftershocks (Weak/Medium).
    *   **Events**:
        *   **Main**: $Base \sim U(6.0, 7.5)$ (Reduced from 8.5)
        *   **Aftershock (Medium)**: $Base \sim U(5.5, 6.5)$
        *   **Aftershock (Weak)**: $Base \sim U(4.5, 5.5)$
    *   **Epicenter Shift**: 여진은 이전 지진의 진앙지 중 하나에서 인접 노드(Neighbor)로 이동하여 발생.

#### **3. 취약도 곡선 (Fragility Curves)**
*   **Method**: `_apply_fragility(states, pga)`
*   **Binary State Model (User Request 2026-02-04)**:
    *   **States**:
        *   `0`: **Safe/Normal** (Passable).
        *   `1`: **Damaged** (Impassable/Risky).
        *   (Legacy `Collapsed` state removed).
*   **Randomized Thresholds**:
    *   엣지별로 임계값에 노이즈($\epsilon \sim N(0, 0.4)$) 추가.
    *   **교량 (Highways)**: Threshold $\approx 6.5g$
    *   **도로 (Normal)**: Threshold $\approx 8.0g$ (Highly Durable)
    *   Logic: $PGA > (Threshold + \epsilon) \rightarrow State=1$

#### **4. 연쇄 파괴 (Cascading Failure)**
*   **Method**: `_apply_cascading_damage`
*   **Sync Logic**:
    *   고가도로(Bridge)가 `Damaged(1)` 상태일 때, 하부 도로(Underpass)도 **80% 확률**로 `Damaged(1)` 상태로 동기화됨.

#### **5. 이전 내용 삭제**
*   Legacy GMPE, Fault Scenarios, Finite Fault Model 등은 모두 제거됨.

## 4. 모델 계층 (Neural Network Layer)

## 4. 모델 계층 (Neural Network Layer)

### 4.1 Manager Model: Markov Decision Process (MDP) Formulation
**`src/models/manager.py`** (`GraphTransformerManager`)

Manager는 글로벌 경로 계획을 수행하며, 이를 **마르코프 결정 과정 (MDP)**으로 정의할 수 있습니다.

#### **MDP Tuple $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma>$**

1.  **State Space ($\mathcal{S}$)**:
    *   $s_t = (G, H_{t})$
    *   $G = (V, E)$: 전체 그래프 정보 (Nodes, Edges, Start, Goal).
    *   $H_{t} = (c_0, c_1, ..., c_t)$: 현재까지 생성된 체크포인트 시퀀스. ($c_0=Start$)
    *   **Representation**: Graph Embedding ($h_G$) + Sequence Embedding via Transformer Decoder ($h_{seq}$).

### 3.2. Agent Parameters
- **Manager (Graph Transformer Manager)**:
  - **Type**: Topology-Aware Graph Transformer (Encoder-Decoder)
  - **Input**:
    - `x`: Node Features [N, 4] (x, y, is_start, is_goal)
    - `edge_index`: Graph Connectivity (Adjacency)
  - **Encoder (Topology-Aware)**:
- **Manager (Graph Transformer Manager)**:
  - **Type**: Hybrid Graph Transformer (GAT + Global Transformer)
  - **Input**:
    - `x`: Node Features [N, 4] (x, y, is_start, is_goal) (Normalized)
    - `edge_index`: Graph Connectivity
    - `target_idx`: [Optional] Target Node Indices [B, L] for Teacher Forcing (Internal Embedding Gathering)
  - **Encoder (Hybrid)**:
    1.  **Local Topology**: `GATv2Conv` (Heads=4) embed local features.
    2.  **Structural Position**: `Laplacian PE` (k=8) projected and ADDED to features.
    3.  **Global Reasoning**: `TransformerEncoder` (3 Layers, 4 Heads) with **Full Self-Attention**.
        *   $r_T = -10.0$ if Fail (Timeout/Invalid).

5.  **Policy ($\pi_\theta$)**:
    *   $\pi_\theta(a_t | s_t) \approx \text{TransformerDecoder}(\text{TransformerEncoder}(G), H_t)$
    *   Encoder: Graph Transformer (Full Self-Attention + Learnable Node Positional Embedding) $\rightarrow$ Node-level Embeddings [B, N, H].
    *   Decoder: Cross-Attention on Node Embeddings + Masked Self-Attention $\rightarrow$ Next Token Logits.

#### **Logit Masking 기법 (Next Token Generation 제약)**
*   **Role**: Pointer Network의 Attention Logits 계산 이후, 잘못된 역방향 탐색이나 무의미한 탐색 공간을 줄여 모델의 수렴 안정성을 극대화합니다.
*   **Method (`generate`) 내부 적용 범위**: `[B, N+1]` 차원의 출력에서 `EOS_TOKEN`(N번째 인덱스)을 제외한 실제 노드 `[:-1]` 영역에만 Float `-inf`를 할당합니다.
1.  **과거 궤적 마스킹 (Visited Mask)**
    *   이미 방문한 노드를 다시 탐색하지 않도록(Loop 방지) 즉각 차단합니다.
2.  **유효 탐색 반경 마스킹 (Radius Mask)**
    *   `APSP` 최단 거리 행렬을 기반으로 홉(hop) 단위의 탐색 반경을 제어합니다.
    *   초근접 노드($< 3$ 홉)는 Worker가 충분히 주행 가능하므로 마스킹 처리하고, 너무 먼 노드($> 10$ 홉)는 Local Pathfinding 정확도를 떨어뜨리므로 배제합니다.
3.  **135도 방향성 마스킹 (Directional Cone Mask)**
    *   현재 노드에서 최종 목적지를 향하는 벡터와 탐색 대상 노드를 향하는 벡터 간 코사인 유사도를 계산합니다.
    *   $135^{\circ}$ 이상 크게 벌어진 후방 궤적 타겟들에 무한대 마스킹을 부여하여, U턴 성향을 방지하고 Goal-Directed 탐색을 유도합니다.

---

### 4.2 Worker Model: Local Execution
**`src/models/worker.py`** (`WorkerLSTM`)

Worker는 Manager가 생성한 Subgoal을 향해 지역적인 이동을 수행합니다.

*   **Role**: Local Pathfinding & Obstacle Avoidance.
*   **Architecture (v2)**:
    *   **Encoder**: `GATv2Conv` × 3 layers + **Residual Connection** + LayerNorm (gradient flow 개선).
    *   **Edge Feature**: `edge_dim=1` ([length] - Phase 1에서는 A*/APSP 학습 신호와 일치하는 유일한 엣지 피처만 사용. Phase 2에서 재난 피처 확장 예정).
    *   **Memory**: `LSTMCell` (Path History).
    *   **Scorer**: **4-head Multi-head Scorer** (독립 MLP 4개 → 평균 앙상블).
    *   **Input**: $x_{local} = [Pos(2), Is\_Robot, Is\_Target, Net\_Dist, Dir(2), Is\_Final\_Target]$. (Total 8 Dims)
    *   **Output**:
        *   `Policy`: Neighbor Selection Probabilities.
        *   `Value`: State Value Estimate (**Critic: 2-Layer MLP** Linear(H,H)->ReLU->Linear(H,1), 2026-04-06).

## 5. 학습 알고리즘 계층 (Trainer Layer)

### 5.1 `src/trainers/pomo_trainer.py`

#### **Class `DOMOTrainer`** (v2 - 2026-02-11 Major Fix)
계층형 POMO (Hierarchical POMO) 알고리즘 구현체.
*   **Workflow**:
    1.  **Generate**: Manager가 $N$개의 다양한 체크포인트 시퀀스(Plan) 생성.
    2.  **Execute**: Worker가 각 시퀀스를 **Gradient 유지 상태**로 순차 실행.
    3.  **Optimize (Vectorized)**:
        *   $N$개의 궤적을 GPU 배치로 병렬 실행 (`batch_size=num_pomo`, Default: **8**).
        *   **[Fix 2026-02-11]** `torch.no_grad()` 제거 → Worker Policy Gradient 완전 복원.
        *   `loss = Policy_Loss + 0.5 * Critic_Loss + Entropy_Bonus`.

#### **Method `train(episodes)` - 6가지 핵심 수정**
1.  **[Fix 1] Worker Gradient 복원**:
    *   기존: `torch.no_grad()` 블록 내 Worker 실행 → gradient 차단됨.
    *   수정: `no_grad` 완전 제거, batch_size 축소(16→8)로 VRAM 대응.
2.  **[Fix 2] Advantage-Reward 일관성 수정**:
    *   Checkpoint Penalty를 Advantage 계산 **전에** 적용하여 불일치 해소.
3.  **[Fix 3] Log-Prob 정규화 (sum→mean)**:
    *   Manager: 유효 토큰 수로 나눈 평균 NLL 사용.
    *   Worker: step_counts로 나눈 평균 log_prob 사용.
    *   효과: Loss 스케일 ~60,000 → ~10 이하로 안정화.
4.  **[Fix 4] Gradient Clipping 추가**:
    *   `nn.utils.clip_grad_norm_(params, max_norm=0.5)` 적용.
5.  **[Fix 5] Curriculum Learning 적용**:
    *   `env.set_curriculum_ratio(ep / (episodes * 0.8))` 호출.
    *   LR: CosineAnnealingWarmRestarts (기존 CosineAnnealing 대체).
    *   Temperature: 초반 1.5 → 후반 0.5 (탐색→수렴).
6.  **[Fix 6] 보상 스케일 균형 조정**:
    *   Step Cost: `1.0` → `0.2` (300스텝 = -60)
    *   Success Bonus: `100` → `50`
    *   Fail Penalty: `50` → `10`
    *   Subgoal Reward: `5.0` (유지)
    *   Checkpoint Penalty: `0.1` → `0.5` per checkpoint

#### **Method `execute_batch_plan`**
*   **Vectorized Worker Execution**:
    *   **[개선]** Gradient 유지 상태에서 Worker 실행 (REINFORCE log_prob 학습).
    *   **Input**: $N$개의 시퀀스 (`[Batch, Max_Len]`).
    *   **Logic**:
        *   `current_subgoal` 도달 여부를 배치 단위로 체크 (`torch.where`).
        *   도달 시 `subgoal_ptr` 증가.
        *   Worker LSTM 및 Policy Net을 배치(`[Batch, Hidden]`)로 한 번에 구동.
        *   **[Fix 2026-03] Dynamic Edge Feature**: `exec_batch_plan` 내부에서 Worker의 `predict_next_hop` 호출 전, `ea = pyg_data.edge_attr[:, 0:1]`를 추출하여 1D 텐서(length만)로 주입. Phase 1에서는 A*/APSP가 length만 사용하므로 다른 피처는 노이즈로 작용하여 제거됨.
        *   **[Fix 2026-03] Worker Gradient 복원 (detach_spatial=False)**: 기존 VRAM 관리를 위해 사용하던 `detach_spatial=True`를 `False`로 변경하여 Worker GATv2 공간 인지 능력이 실제로 역전파(Backprop)를 통해 학습되도록 강제함. 이에 따라 기본 `--batch_size`를 8에서 4로 축소.
        *   **[Fix 2026-03] Loop Penalty 및 Critic Window 완화**: 지나치게 엄격했던 `LOOP_PENALTY_SCALE`을 0.5에서 0.05로 대폭 완화하여 초반 탐색 중 조기 종료(Timeout)되는 현상을 방지. 또한 `CRITIC_WINDOW`를 50에서 `MAX_TOTAL_STEPS`로 상향 조정하여 Bias 편향 제거.
        *   **[Fix 2026-03] Manager Target Padding Masking**: `target_seq_emb` 추출 시 패딩 토큰(PAD)에 대해 잘못된 임베딩(Node 0)이 주입되지 않도록 `0.0`으로 명시적 마스킹 처리.
    *   **VRAM 관리**: batch_size 축소(64→8→4)로 gradient 그래프 유지에 따른 메모리 증가 대응.
    *   **[New 2026-03-11] Diagnostic Logger (`--debug` 플래그)**:
        *   `train_rl.py --debug` 활성화 시 200 에피소드마다 상세 진단 패널 출력.
        *   **보상 분해**: R1(Shaping), R2(Subgoal), R3(Goal), R4(Efficiency), P1(Step), P2(Loop), P3(Fail) 각 요소의 배치 평균값.
        *   **Worker 진단**: 서브골 도달률, 평균 액션 Entropy, 평균 스텝 수.
        *   **Critic 진단**: 초기 Value 예측(`V0`), MSE Loss, Advantage 평균/표준편차.
        *   **Gradient Norm**: Manager/Worker 파라미터별 L2 Gradient Norm.
        *   **Manager 진단**: Entropy, NLL, Policy Loss, Critic Loss.
        *   **데이터 흐름**: `execute_batch_plan()` → `self._last_diag` (dict) → `train()` 에서 읽어 출력.
    *   **[Fix 2026-03] Worker 병목 해결 및 학습 안정화**:
        *   **보상 전면 재설계 (PBRS + 7요소)**: R1 Shaping이 무한 누적 증폭되어 학습을 방해하던 치명적 결함을 수학적으로 증명된 **Potential-Based Reward Shaping (PBRS)** 로 교체. `POTENTIAL_SCALE(15.0)`, `SUBGOAL_BASE(5.0) + SCALE(10.0)`, `OPTIMALITY_BONUS(3.0)`, `MILESTONE_25/50/75(3.0/5.0/8.0)`, `EXPLORATION_BONUS(0.3)`, `TIME_PRESSURE`, `GOAL_REWARD(20.0)`, `FAIL_PENALTY(-5.0)`. APSP 매트릭스를 활용해 Worker에게 정확한 가이드 제공.
        *   **Gradient Clipping 완화**: `max_norm`을 0.5에서 2.0으로 상향. Worker(190만+), Manager(760만+) 파라미터 규모에서 0.5는 매 에피소드 100% 잘림 발생.
        *   **Advantage 정규화**: `(adv - mean) / (std + 1e-8)` 적용. Advantage Std가 3~20을 오가며 Policy Gradient를 불안정하게 만들던 문제 해결.
        *   **LR 스케줄러 교체**: `CosineAnnealingWarmRestarts` → `CosineAnnealingLR` (단조 감소). 주기적 LR 재시작이 SL 사전학습 지식을 파괴하던 Catastrophic Forgetting 방지.
    
#### **Method `_validate_worker` (v5 Fix) [New]**
*   **VRAM Leak Fix**: 검증(Validation) 루프에서도 Worker의 LSTM hidden state (`h`, `c`) 텐서가 윈도우 스텝을 넘어갈 때마다 연산 그래프(Computational Graph)를 계속 누적하여 VRAM이 1GB 이상 증가하는 누수 현상이 발견되었습니다.
*   **Solution**: `h = h_new.detach()`, `c = c_new.detach()` 코드를 검증 루프 내에 명시적으로 추가하여, 각 BPTT 스텝이 끝날 때마다 과거 그래프 노드와의 연결을 물리적으로 해제(Garbage Collection 유도)함으로써 메모리 누수를 완전히 해결했습니다.

---


## 6. 계층적 모방 학습 (Hierarchical Imitation Learning) - [New]

**목표**: `Anaheim` 등 대규모 맵에서의 효율적 경로 탐색을 위해, 전문가(Expert - A*)의 경로를 계층적으로 학습합니다.

### 6.1 `src/data/generate_expert.py` (Phase 0)
대규모 전문가 데이터 생성기입니다.
*   **Checkpoint Selection**: **Full A* Path**.
    *   기존의 K-Means 군집화 방식은 제거됨.
    *   Manager는 전문가(A*)가 이동한 **전체 경로(Sequence of Nodes)**를 초기 정답으로 모방 학습함.
    *   RL 단계에서 보상(Reward)을 통해 불필요한 노드를 생략(Skip)하며 최적 체크포인트 전략을 자가 발전시킴.
*   **Trajectory Generation**: 50,000 쌍의 $(O, D)$에 대해 A* 경로 생성.
*   **Vectorization & Optimization (2026-02-07)**:
    *   **Direct Tensor Saving**: Pickle 객체 생성 오버헤드를 제거하고, `torch.save`를 통해 `.pt` 포맷으로 직접 저장.
    *   **Time Complexity**: Worker 라벨링 로직을 $O(N^2)$에서 **$O(N)$**으로 최적화.
    *   **File Structure**:
        *   `manager_data.pt`: `{start, goal, sequences}` Tensors
        *   `worker_data.pt`: `{curr, target, next}` Tensors (Structure of Arrays)

### 6.2 모델 아키텍처 (Phase 1)
*(중략)*

### 6.4 `train_sl.py` (Phase 2 - Sequence Pre-training)

#### **`src/data/segment_loader.py`**
*   **Role**: 전처리된 Tensor 데이터를 로드하여 PyTorch Dataset으로 관리.
*   **Structure of Arrays (SoA)**:
    *   기존 `List[Dict]` 방식(AoS)의 메모리 오버헤드를 제거하기 위해, 속성별로 분리된 Tensor(`start_nodes`, `goal_nodes` 등)를 유지.
    *   `__getitem__` 시점에 필요한 인덱스의 스칼라 값만 추출하여 가볍게 반환.
*   **Direct Loading**: `__init__`에서 `.pt` 파일(Dictionary of Tensors)을 그대로 받아 할당. 복사/변환 과정 없음.

#### **`train_sl.py` (Training Logic)**
- **Optimizer**: Adam
  - **Manager LR**: `5e-4` (Scheduler: `ReduceLROnPlateau`, Patience=5)
  - **Worker LR**: `1e-4` (Scheduler: `CosineAnnealingLR`)
- **Loss Function**:
  - Manager: `0.5 * CrossEntropy + 0.5 * KL_Divergence` (Soft Label)
    - **[Optimization] Soft Label Distance**: 유클리디안 물리적 거리 대신 `APSP` 네트워크 최단 거리를 사용하여 지도 제약 조건의 정확도를 향상시킴.
    - **[Optimization] Temperature Sync**: 훈련(1.0)과 검증(0.5) 시 달랐던 Soft Label 분포 온도를 `1.0`으로 일관되게 동기화.
  - Worker: `CrossEntropy`
- **Worker Training Frequency**: 1/5 Epochs (Manager 5 : Worker 1)
  - *Reason*: Worker converges much faster than Manager. To prevent overfitting and save time, Worker is trained less frequently.
*   **DataLoader & GPU Optimization**:
    *   **Workers**: `4` (Memory/Speed Sweet-spot).
    *   **Pin Memory**: `True` (GPU Transfer 가속).
    *   **Global APSP Caching**: `apsp_matrix`의 `.to(device)` 전송을 에폭 무한 루프 밖으로 빼내어 `apsp_device_global`로 단일화 (PCIe 병목 제거).
    *   **Simulation Caching**: 평가/시뮬레이션 루프 내부에서 매 스텝 호출되던 `.to(device)` 이동 연산을 루프 외부로 빼내어 정적 텐서(Static Tensors) 공간에 미리 병렬 캐싱 (메모리 대역폭 절약).
*   **Manager Training (Teacher Forcing)**:
    *   **Input**: `[SOS, C1, C2, ..., PAD]`
    *   **Target**: `[C1, C2, ..., EOS, PAD]`
    *   **Loss**: `CrossEntropyLoss(ignore_index=-100)`.
    *   **Dynamic Token Handling (Robustness)**:
        *   **SOS**: `Num_Nodes` (Virtual Start) - Preprended to Input.
        *   **EOS**: `Num_Nodes` (Virtual End) - Appended to Target.
        *   **Embeddings**: `manager.eos_token_emb` used for EOS positions.
        *   **Soft Label**: Applied only to real node targets ($< Num\_Nodes$), EOS gets 0 probability.
*   **Worker Training**:
    *   **Input**: `[x, y, is_curr, is_tgt, dist, dir_x, dir_y]` (7 Channels).
    *   **Loss**: `CrossEntropyLoss` (Target: Next Hop Index) + 평균 BPTT 역전파.
    *   **Optimization**: Adam (LR=1e-3).
    *   **다중 스텝 시퀀스 훈련 (Multi-step Unrolling) 및 벡터화 연산**:
        *   A* 전문가가 방문한 경로 배열 전체(`[Cur, Tar, Nxt]`)를 하나의 시퀀스로 묶어 처리.
        *   **[Optimization] Vectorized CrossEntropy Loss**: 파이썬 `for` 문으로 스텝별 로스를 묶어 누적하던 오버헤드를 제거, `wkr_criterion(scores_view, labels_tensor)` 기반 단일 매트릭스 연산으로 로스를 계산합니다.
        *   **[Optimization] Feature Vectorization**: 기존 파이썬 반복문을 통해 `dummy_x` 및 `flags`를 덧붙이던 O(N^2) 병목을 제거. 배치 내 유효 그래프의 인덱스를 벡터 연산(`torch.where`)으로 추출해 전체 노드 텐서에 한 번에 값어사인.
        *   **[Optimization] Native Edge Padding**: 서브 그래프 단위의 `sub_edge_index` 동적 재생성 비용을 제거하고 원본 `batch.edge_index`를 통과시키되 패딩된 노드 특성을 0으로 처리하는 방식으로 치환하여 속도 극대화.
        *   윈도우(Window=5) 기반 Truncated BPTT 역전파를 수행하여 Hidden State 메모리를 최적화.
    *   **Teacher Forcing Ratio Decay**:
        *   Autoregressive 성능을 끌어올리기 위해 $TF=1.0$에서 점진적으로 강요 비율을 축소. 예측 성능 붕괴를 막기 위해 하한선(Lower Bound)을 **0.1**로 설정.
    *   **Accuracy (Acc)**:
        *   **Definition**: Next Token Prediction Accuracy (Teacher Forcing).
        *   **Formula**: $\frac{\sum (\text{Pred} == \text{Target})}{\sum \text{Total Valid Tokens}}$
        *   **Meaning**: 매 타임스텝마다 정답(A* 경로의 다음 노드)을 정확히 맞춘 비율. 초기에는 탐색 공간이 넓어(416개 노드) 낮게 시작하지만, 학습이 진행될수록 90% 이상으로 상승함.
*   **Checkpoint & Metadata Saving**:
    *   **[Optimization] Unified Dictionary Save**: RL 파인튜닝 시 `max_dist`나 `num_nodes` 불일치 휴먼 에러를 방지하기 위해 `manager_state`, `worker_state` 및 환경 매개변수를 묶어 단일 파일(`model_sl_final.pt`)에 저장합니다.

#### **`train_sl.py` LLD (Low-Level Design) - Training Loops**

*   **Multi-Threading & Sequential Toggle Architecture (2026-04-22)**:
    *   **Structure**: `run_sequential_pipeline()` (기본값) 및 `args.parallel` 옵션 지정 시 `run_manager_pipeline()`, `run_worker_pipeline()` 두 개의 독립 쓰레드로 분리 실행.
    *   **Sequential Mode (`--parallel` 미지정 시)**: 
        *   동일 그래픽카드에서 이전 방식과 동일하게 Manager 1 epoch -> Worker 1 epoch 동시 루프(Synchronized) 진행으로 안정성 유지.
    *   **Parallel Mode (Dual-GPU `--parallel` 지정 시)**:
        *   Manager는 **GPU 0**(`cuda:0`)에, Worker는 **GPU 1**(`cuda:1`)에 각각 독립적으로 적재됩니다.
        *   공유 객체 사용 및 메모리 락(Lock) 없이 완벽히 분리된 연산 풀에서 그래픽카드를 100% 한계치까지 이중(2-way)으로 사용하여 학습 시간을 획기적으로 절반으로 단축시킵니다 (GPU 병목 완전 해소).
        *   Worker의 Validation 검증 등 교차 참조 시 `worker_frozen` 전역 플래그를 통해 Early Stopping 상태를 제한적으로 동기화.
        *   터미널 출력 충돌 방지를 위해 `tqdm(position=0)`과 `position=1`을 할당.

*   **Manager Training Loop (Vectorized target assignment)**:
    *   **Data Flow**:
        1. `raw_targets` : Shape `[B, L]` (Valid nodes + PAD=-100)
        2. `lengths` : Shape `[B]` (Sequence Length per batch)
        3. `final_targets` : Shape `[B, L+1]` (Allocated with PAD_VAL)
        4. Vectorized Assignment: `valid_mask` (`[B, L]`)을 생성하여 복사, 이후 `lengths` 인덱스에 `EOS_IDX` 할당.
    *   **Optimization**: Python For-loop를 완전히 배제하여 CPU-GPU Memory Transfer 동기화 지연을 차단함.

*   **Worker Training Loop (Vectorized Dynamic Batch Shrinking)**:
    *   **Variables**:
        *   `h`, `c` : Shape `[B, Hidden_Dim]` (단일 텐서 LSTM State, 활성 그래프만 슬라이싱하여 사용)
        *   `active_idx` : Shape `[K]` (현재 스텝에서 시퀀스가 끔나지 않은 그래프 인덱스)
        *   `compact_edge_index` : Shape `[2, K*E]` (K개 그래프만의 엣지 인덱스)
        *   `worker_in` : Shape `[K*N, 7]` (K개 그래프만의 입력 피처)
        *   `c_seqs_pad`, `t_seqs_pad` : Shape `[B, Max_Seq_Len]` (CPU-GPU 블로킹 방지용 패딩 텐서)
    *   **Pseudo-code (Logic)**:
        ```python
        # 루프 밖에서 단 한 번만 구조 준비 및 패딩 텐서 생성
        targets_padded = pad_sequence(targets)
        node_coords_per_graph = batch.x.view(B, N, 2)
        base_edges = pyg_data.edge_index  # 관형(単体) 그래프 1개분
        h, c = zeros(B, Hidden_Dim)

        for step in range(Max_Seq_Len):
            active_idx = (step < seq_lens).nonzero()  # K개
            K = len(active_idx)

            # K개 그래프만의 컴팩트 GNN 입력 조립 (B개 -> K개 축소)
            compact_coords = node_coords_per_graph[active_idx]  # [K*N, 2]
            compact_edge_index = base_edges + offsets(K)         # [2, K*E]
            worker_in = Build_Features(compact_coords, ...)     # [K*N, 7]

            # K개만 GATv2 Forward (GPU 연산량 선형 감소!)
            scores, h_next_k, c_next_k = GATv2(worker_in, compact_edge_index,
                                                h[active_idx], c[active_idx])

            # 은닉 상태 원본 텐서에 매핑 (BPTT 유지)
            h_new = h.clone()
            h_new[active_idx] = h_next_k
            h = h_new

            # Loss (K개만 계산, 마스킹 불필요)
            step_loss = CrossEntropy(scores.view(K, N), targets_k)
        ```
    *   **Optimization Details**:
        *   **핵심**: 매 스텝마다 GATv2에 릤개되는 노드 수가 `B*N` → `K*N`으로 동적 축소. 시퀀스 후반부로 갈수록 K→0에 수렴하여 FLOPs가 기하급수적으로 감소.
        *   PyG의 `subgraph()` 같은 무거운 유틸리티 없이, `base_edges + offsets` 단순 덧셈만으로 컴팩트 엣지 인덱스를 마이크로초 단위로 조립.
        *   `.item()`, `.tolist()` 같은 CPU-GPU 동기화 함수 완전 제거 (`pad_sequence` + 텐서 슬라이싱).
        *   Loss 계산 시 K개만 대상이므로 마스킹 불필요, 방어적 분모(`+1e-8`) 불필요.

*   **Simulation / Evaluation Loop (Pre-allocation)**:
    *   **Data Flow**:
        *   `is_current`, `is_target` : Shape `[Num_Nodes, 1]` 텐서를 Loop 밖에서 사전 할당.
        *   루프 내부에서는 새로운 메모리를 요청(malloc)하지 않고 `.zero_()` 메소드(In-place)로 텐서를 초기화 후 값 삽입. 결과적으로 시뮬레이션 평가 속도 대폭 상승 및 단편화 방지.
        *   **[Fix] Edge Attributes in Evaluation**: Worker Accuracy 검증(Evaluation) 루프에서 `edge_attr_device` 선언 전 참조 오류(UnboundLocalError) 방지를 위해, 전역으로 로드된 `pyg_data.edge_attr.to(device)`를 직접 주입하여 메모리 참조 순서를 안정화.

---

## 7. 통합 평가 CLI (	ests/evaluate.py + 	ests/eval_core.py)

기존 7개 분산 스크립트를 **2개 파일**로 통합한 단일 진입점.

### 7.1 파일 구조

| 파일 | 역할 |
|------|------|
| 	ests/eval_core.py | 공유 롤아웃 엔진, 환경/모델 로드, 배치 평가, A* 벤치마크, 경로 분석 |
| 	ests/evaluate.py | 통합 CLI (7개 서브커맨드) + 학습 대시보드 + 롤아웃 진단 시각화 |

### 7.2 서브커맨드

| 서브커맨드 | 기존 스크립트 | 기능 |
|-----------|-------------|------|
| dashboard | isualize_result.py | 학습 로그 파싱, 대시보드, 롤아웃 진단 |
| worker | eval_worker_batch.py | Worker 배치 성능 평가 (SR + PLR) |
| joint | eval_joint_rollout.py | Manager + Worker 조인트 롤아웃 |
| compare | compare_models.py | SL vs RL 모델 비교 |
| memorization | eval_worker_memorization.py | Worker 암기 vs 일반화 진단 |
| physics | isualize_physics.py | 재난 물리 (지진) 시각화 |
| paper | generate_paper_figures.py | 논문용 Figure/Table 자동 생성 |

### 7.3 실행 예시

`ash
python tests/evaluate.py dashboard --run-dir logs/rl_phase1_apte
python tests/evaluate.py worker --checkpoint logs/.../final.pt --trials 100
python tests/evaluate.py compare --sl logs/sl/.../model_sl_final.pt --rl logs/rl/.../final.pt
python tests/evaluate.py paper --map Anaheim --eval-episodes 100
python tests/evaluate.py paper_full --joint-ckpt [path] --worker-ckpt [path] --episodes 500
python tests/evaluate.py regen --joint-ckpt [path] --worker-ckpt [path] --map Goldcoast
`

### 7.4 eval_core.py 핵심 API

`python
setup_env(map_name, device) -> DisasterEnv
load_checkpoint(path, device, load_manager) -> (WorkerLSTM, Optional[Manager], Dict)
run_worker_rollout(worker, env, start, goal, max_steps, temperature) -> Dict
run_joint_rollout(worker, manager, env, start, goal, max_steps) -> Dict
evaluate_worker_batch(worker, env, num_episodes, label, seed) -> Dict
benchmark_astar(env, num_queries) -> Dict
compute_path_overlap(worker_path, optimal_path) -> Dict
`

## 10. 	ests/eval_core.py 업데이트 (Joint Rollout 및 GPU 타이머 추가)
- **Module Target**: 	ests/eval_core.py
- **추가 기능**: Manager와 Worker를 동시에 롤아웃하여 성능을 측정하는 evaluate_joint_batch 추가.
- **수정본 Signature**: def run_joint_rollout(worker, manager, env, ... measure_time: bool = False) -> Dict
- **Data Flow 변화**: Manager 텐서 처리 후 `torch.cuda.synchronize()`를 호출하여 GPU 기반 HRL Latency를 정확하게 분리 측정.

---

## 11. Worker v4 아키텍처 및 State Space 전면 개편 (2026-04-23)
- **Module Target**: `src/models/worker.py`, `train_sl.py`, `src/trainers/worker_nav_trainer.py`
- **핵심 목표**: 특정 맵에 과적합(Overfitting)되는 문제를 해결하고, POMDP 한계(방향 감각 상실)를 극복하기 위해 Node/Edge Feature를 전면 재설계.

### 11.1 Node Feature 개편 (v4 State: 8-Dim)
- **제거된 피처**: `net_dist`, `dir_x`, `dir_y` (유클리드 방향 및 국소 거리)
- **신규 8-Dim 구조**:
  - `0`: `is_curr` (현재 위치)
  - `1`: `is_subgoal` (Manager가 할당한 목표 노드)
  - `2`: `is_final_goal` (에피소드의 최종 도착 노드)
  - `3`: `hop_to_subgoal` (각 노드에서 subgoal까지의 정규화된 홉 거리)
  - `4`: `hop_to_final` (각 노드에서 최종 도착지까지의 정규화된 홉 거리)
  - `5, 6`: `global_heading_x, y` (각 노드에서 최종 도착지를 향하는 유클리드 방향 단위 벡터)
  - `7`: `time_to_go` (1.0 - t/T, 스텝 진행률 역산)
- **기대 효과**: 에이전트가 맵의 절대 좌표나 형태에 얽매이지 않고, "최종 목적지 방향성"과 "서브골까지의 위상학적 거리"를 기반으로 최적화된 경로를 선택 (Zero-shot Generalization 향상).

### 11.2 GNN 아키텍처 다이어트 (GraphNorm & Single-head)
- **`LayerNorm` → `GraphNorm` 교체**: GATv2 내부에서 발생하던 Scale-induced Softmax Saturation (엣지가 쏠려 attention이 붕괴되는 현상)과 Over-smoothing을 방어.
- **Scorer Head 최적화**: 4개였던 Multi-head Scorer를 단일(Single) Head로 축소하여 다중공선성(Multicollinearity) 위험을 제거하고, 연산 오버헤드 감소.
- **Global Z-score Normalization (Edge)**: 기존 Min-Max 방식에서 `(X - Mean) / Std` (최소 1e-8 clamp)로 Edge Feature(거리, 속도, 용량) 스케일링을 변경하여 공변량 편이 방지.

### 11.3 RL 훈련 파이프라인 개편 (예정)
- **PBRS 보상 개편**: 유클리드 거리 기반 보상을 `hop_dist` 기반으로 전환 준비. 
- **Trainer PPO 전환**: 기존 A2C(REINFORCE + Baseline) 방식의 `WorkerNavTrainer` 구조를 단일 Transition 수집 후 Ratio 클리핑하는 PPO 아키텍처로 고도화하여 Policy Collapse 문제 해결.
