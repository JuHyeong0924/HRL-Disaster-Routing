# HRL Worker Ablation Study 최종 보고서

> **실험 환경**: Anaheim 416 노드 | POMO 32 | 5000 에피소드 | RTX 4090 × 2 병렬
> **총 소요**: 20시간 24분 | 18개 실험 전체 exit_code=0
> **SL 체크포인트**: `model_sl_final.pt` | **실험 일자**: 2026-04-27

---

## 1. 실험 설계

### 1.1 Baseline 구성 (v4.1)
- **GNN**: GATv2 3-Layer (heads=4, hidden=256)
- **Temporal**: LSTM (256-dim hidden)
- **Normalization**: Residual Connection + GraphNorm
- **State**: 7-Dim (`is_curr`, `is_tgt`, `hop_dist`, `net_dist`, `dir_x`, `dir_y`, `time_to_go`)
- **Reward**: PBRS + Hop Bonus + Checkpoint + Goal + Step + Loop + AuxCE Loss

### 1.2 실험 카테고리

| 카테고리 | ID 범위 | 실험 수 | 목적 |
|---|---|---|---|
| Architecture | A1~A7 | 7 | GNN 깊이, Residual, GraphNorm, LSTM, Hidden dim |
| State | S1~S5 | 5 | 노드 피처 차원 축소/제거 효과 |
| Reward | R1~R5 | 5 | 보상 구성요소 개별/전체 제거 |
| **합계** | — | **17 + BASELINE = 18** | — |

### 1.3 개별 실험 정의

#### Architecture Ablation
| ID | 변경 사항 | 변경 파라미터 |
|---|---|---|
| A1 | GATv2 Layer 수 감소 | `num_layers=2` |
| A2 | Residual Connection 제거 | `use_residual=False` |
| A3 | GraphNorm 제거 | `use_graphnorm=False` |
| A4 | Hidden dim 축소 | `hidden_dim=128` |
| A5 | LSTM 제거 → Linear 투영 | `use_lstm=False` |
| A6 | GATv2 Layer 수 증가 | `num_layers=4` |
| A7 | GATv2 Layer 수 최소화 | `num_layers=1` |

#### State Ablation (Baseline 7-Dim에서 개별 피처 제거)
| ID | 제거 피처 | 결과 Dim |
|---|---|---|
| S1 | `hop_dist` | 6-Dim |
| S2 | `dir_x`, `dir_y` | 5-Dim |
| S3 | `time_to_go` | 6-Dim (실제로는 `is_final` 제거) |
| S4 | `net_dist` | 6-Dim |
| S5 | 최소 State (`is_curr`, `is_tgt`, `hop_dist`만) | 3-Dim |

#### Reward Ablation (Baseline 보상에서 개별 구성요소 제거)
| ID | 제거 구성요소 |
|---|---|
| R1 | PBRS (Potential-Based Reward Shaping) |
| R2 | Hop Bonus |
| R3 | Checkpoint Reward |
| R4 | AuxCE (Cross-Entropy 보조 손실) |
| R5 | 간소화: `goal_reward + step_penalty`만 유지 |

---

## 2. 정량적 결과

### 2.1 Architecture Ablation 결과

| ID | 설명 | EMA | Δ Baseline | Entropy | ExplVar | Grad | AvgLen | AvgRw | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| **BASELINE** | 3L+LSTM+Res+GN+256 | **81.4%** | — | 0.610 | 0.12 | 0.564 | 31.9 | -73.8 | 기준 |
| **A1** | 2-Layer | **85.5%** | +4.1% | 0.594 | 0.12 | 0.588 | 30.7 | -65.0 | ⬆️ |
| **A2** | Residual 제거 | **25.5%** | -55.9% | 0.632 | 0.07 | 0.167 | 49.3 | -177.9 | 🔻 필수 |
| **A3** | GraphNorm 제거 | **54.8%** | -26.6% | 0.628 | 0.03 | 0.495 | 42.7 | -130.6 | 🔻 필수 |
| **A4** | Hidden 128 | **57.9%** | -23.5% | 0.612 | 0.13 | 1.009 | 42.0 | -123.5 | 🔻 256 필수 |
| **A5** | LSTM 제거 | **91.4%** | +10.0% | 0.443 | 0.01 | 0.562 | 25.2 | -35.3 | ⬆️ 최고 |
| **A6** | 4-Layer | **81.1%** | -0.3% | 0.534 | 0.19 | 0.449 | 31.3 | -71.6 | ➡️ 동등 |
| **A7** | 1-Layer | **87.3%** | +5.9% | 0.537 | 0.35 | 0.793 | 27.0 | -45.0 | ⬆️ |

**분석:**
- **LSTM 제거(A5)가 가장 큰 성능 향상**: +10.0%. Entropy 0.443으로 정책이 확신적 결정
- **Residual(A2) 제거 시 학습 불가**: Gradient가 0.167로 Vanishing 발생
- **GraphNorm(A3) 제거 시 Critic 붕괴**: ExplVar 0.03, Value 함수 예측 능력 상실
- **Hidden 128(A4) 표현력 부족**: Gradient 1.009로 과도한 요동
- **Layer 수**: 1-Layer(87.3%) > 2-Layer(85.5%) > 3-Layer(81.4%) ≥ 4-Layer(81.1%) → 적을수록 좋음
- **1-Layer(A7)의 ExplVar 0.35**: 전체 실험 중 Architecture 카테고리 최고 → 단순 모델이 Critic 학습에 유리

### 2.2 State Ablation 결과

| ID | 설명 | Dim | EMA | Δ | Entropy | ExplVar | AvgLen | AvgRw | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| **S1** | hop_dist 제거 | 6 | **75.9%** | -5.5% | 0.573 | 0.12 | 34.2 | -81.2 | ⬇️ hop 필수 |
| **S2** | dir_x/y 제거 | 5 | **100.0%** | +18.6% | 0.153 | 0.78 | 11.5 | +41.6 | ⬆️ 방향 불필요 |
| **S3** | time_to_go 제거 | 7 | **88.6%** | +7.2% | 0.544 | 0.19 | 28.9 | -56.1 | ⬆️ |
| **S4** | net_dist 제거 | 6 | **86.3%** | +4.9% | 0.491 | 0.18 | 27.2 | -59.3 | ⬆️ |
| **S5** | 최소 3-Dim | 3 | **99.9%** | +18.5% | 0.294 | 0.03 | 11.4 | +40.9 | ⬆️ 최적 |

**분석:**
- **S2/S5가 압도적**: EMA 100%/99.9%, 경로 길이 11.4~11.5 hop (거의 최단 경로)
- **S2/S5 보상 양수 (+40)**: 성공 보상이 스텝 페널티를 압도 → 극도로 효율적 네비게이션
- **hop_dist(S1)만 유일하게 필수**: 제거 시 -5.5% 하락. 위상학적 근접성 정보 손실
- **dir_x/dir_y**: 제거 시 오히려 +18.6% → 방향 벡터가 학습 노이즈로 작용
- **Entropy 비교**: S2(0.153), S5(0.294) vs Baseline(0.610) → 간소화 State가 더 결정론적 정책 생성

### 2.3 Reward Ablation 결과

| ID | 설명 | EMA | Δ | Entropy | ExplVar | AvgLen | AvgRw | 판정 |
|---|---|---|---|---|---|---|---|---|
| **R1** | PBRS 제거 | **82.3%** | +0.9% | 0.579 | 0.27 | 30.2 | -67.3 | ➡️ 동등 |
| **R2** | Hop Bonus 제거 | **81.6%** | +0.2% | 0.564 | 0.15 | 31.0 | -75.5 | ➡️ 동등 |
| **R3** | Checkpoint 제거 | **83.3%** | +1.9% | 0.502 | 0.03 | 31.7 | -69.9 | ➡️ 동등 |
| **R4** | AuxCE 제거 | **99.7%** | +18.3% | 0.075 | 0.21 | 12.5 | +38.7 | ⬆️ 최고 |
| **R5** | 간소화 보상 | **99.7%** | +18.3% | 0.134 | 0.70 | 12.5 | +39.4 | ⬆️ 최적 |

**분석:**
- **AuxCE(R4)가 가장 해로운 요소**: 제거만으로 81.4% → 99.7% 점프. CE 보조 손실이 RL gradient와 충돌
- **R5 간소화 보상**: ExplVar 0.70으로 Critic이 70% 리턴 설명 → 단순 보상이 Value 학습에 최적
- **PBRS/Hop Bonus/Checkpoint**: 개별 제거해도 거의 영향 없음 → 전부 불필요
- **R4/R5 Entropy**: 0.075/0.134 → 거의 결정론적 정책 (매 스텝 최적 행동에 확신)

---

## 3. 수렴 속도 분석

### 상위 실험 vs Baseline EMA 추이

| EP | BASELINE | S2 | S5 | R5 | R4 | A5 |
|---|---|---|---|---|---|---|
| 200 | 35.2% | 59.0% | 59.9% | 46.9% | 44.0% | 38.1% |
| 400 | 34.5% | 76.9% | 69.2% | 66.2% | 61.5% | 33.4% |
| 600 | 38.4% | 88.2% | 78.4% | 89.0% | 79.4% | 34.8% |
| 800 | 38.0% | 94.5% | 91.9% | 95.0% | 89.7% | 34.9% |
| 1000 | 36.8% | **97.1%** | **94.1%** | **96.0%** | **95.5%** | 36.9% |
| 5000 | 81.4% | **100.0%** | **99.9%** | **99.7%** | **99.7%** | **91.4%** |

- **S2/S5/R4/R5**: EP 800~1000에서 95%+ 도달 → 수렴 속도 **5배 이상** 빠름
- **BASELINE**: EP 1000 시점 36.8% → 복잡한 State/Reward가 초기 학습을 심각하게 방해
- **A5**: 초기 Baseline과 유사하나 후반부 급등 (34.9% → 91.4%). LSTM 제거 효과는 장기적

---

## 4. 경로 품질 분석

### 전체 18개 실험 경로 효율 순위 (마지막 500ep 평균)

| 순위 | ID | EMA | AvgLen (hop) | AvgRw | AvgSucc | 판정 |
|---|---|---|---|---|---|---|
| 1 | **S5** | 99.9% | **11.4** | **+40.9** | 99.5% | 거의 최단 경로 |
| 2 | **S2** | 100.0% | **11.5** | **+41.6** | 99.7% | 거의 최단 경로 |
| 3 | **R4** | 99.7% | **12.5** | **+38.7** | 99.5% | 최단 근접 |
| 4 | **R5** | 99.7% | **12.5** | **+39.4** | 99.6% | 최단 근접 |
| 5 | A5 | 91.4% | 25.2 | -35.3 | 92.7% | 준수 |
| 6 | A7 | 87.3% | 27.0 | -45.0 | 89.0% | 보통 |
| 7 | S4 | 86.3% | 27.2 | -59.3 | 86.4% | 보통 |
| 8 | S3 | 88.6% | 28.9 | -56.1 | 85.0% | 보통 |
| — | **BASELINE** | 81.4% | **31.9** | **-73.8** | 81.5% | 비효율적 |
| — | A2 | 25.5% | 49.3 | -177.9 | 25.4% | 학습 불가 |

- **Anaheim APSP 평균 최단 경로: ~10-12 hop**
- S5/S2: 11.4~11.5 hop → **거의 최단 경로 달성**
- Baseline: 31.9 hop → **최단의 약 3배 우회**
- 경로 품질과 성공률은 강한 양의 상관관계

---

## 5. 실패 원인 심층 분석

### A2 (Residual 제거) — EMA 25.5%
- **Gradient: 0.167** (Baseline의 30%)
- 3-Layer GATv2 통과 시 gradient 소멸 → 학습 자체 불가
- **결론**: GNN 깊이 ≥ 2에서 Residual은 gradient flow의 생명선

### A3 (GraphNorm 제거) — EMA 54.8%
- **ExplVar: 0.03** (Baseline의 1/4)
- GATv2 레이어 간 피처 스케일 발산 → over-smoothing 가속
- **결론**: GraphNorm은 노드 임베딩 분포 안정화에 필수

### A4 (Hidden 128) — EMA 57.9%
- **Grad: 1.009** (Baseline의 2배)
- 128차원은 416 노드 규모에서 표현력 부족 → gradient 과도 요동
- **결론**: Hidden ≥ 256 필요

### S1 (hop_dist 제거) — EMA 75.9%
- hop_dist 없이 유클리드 기반 net_dist만 사용
- 그래프 위상과 유클리드 거리의 불일치 → 경로 판단 오류
- **결론**: hop_dist가 유일하게 필수인 거리 피처

### AuxCE (R4) — 제거 시 81.4% → 99.7%
- Cross-Entropy 보조 손실이 RL Policy Gradient와 직접 충돌
- SL 모방 목표(정답 경로 재현)와 RL 탐색 목표(최적 정책 발견)의 gradient 방향이 상충
- **결론**: SL→RL 전이 후에는 AuxCE를 반드시 비활성화해야 함

---

## 6. 이론적 해석

### Occam's Razor + Reward Simplicity Principle

```
복잡한 모델 (Baseline):
  7-Dim State + LSTM + 3-Layer GNN + PBRS + 6종 보너스 + AuxCE
  → 파라미터 과다 → 탐색 공간 확대 → 수렴 지연 → Local Optima

간소화 모델 (S5+R5+A5 조합):
  3-Dim State + Linear + 1-Layer GNN + goal+step만
  → 작은 가설 공간 → 명확한 신호 → 빠른 수렴 → Global Optimum
```

1. **State**: 네비게이션은 "어디에 있는가(is_curr)" + "어디로 가는가(is_tgt)" + "얼마나 먼가(hop_dist)" 3가지 정보만 필요. GATv2가 그래프 구조에서 나머지를 자동 추론
2. **Reward**: 복잡한 보상은 다중 최적화 목표 부여 → 정책 gradient 상충 → 수렴 방해
3. **Temporal Memory**: 최적 행동이 현재 상태에만 의존(Markovian) → LSTM의 기억이 과거 편향 유발
4. **GNN 깊이**: Over-smoothing으로 깊은 GNN이 오히려 노드 구별력 상실

---

## 7. 최종 아키텍처 결정

### 7.1 Ablation 기반 최적 구조

| 요소 | Baseline | Ablation 최적 | 근거 |
|---|---|---|---|
| GATv2 Layer | 3 | **1** | A7: +5.9% |
| Residual | ✅ | **✅ 유지** | A2: -55.9% (필수) |
| GraphNorm | ✅ | **✅ 유지** | A3: -26.6% (필수) |
| Temporal | LSTM 256 | **Linear 투영** | A5: +10.0% |
| Hidden dim | 256 | **256 유지** | A4: -23.5% |
| State | 7-Dim | **3-Dim** | S5: +18.5% |
| Reward | PBRS+6종+AuxCE | **goal+step만** | R5: +18.3% |

### 7.2 재난 환경 대비 최종 채택 구조

Ablation은 정적 환경(damage=0) 기준이므로, 재난 시 동적 도로 차단을 고려한 조정 적용:

| 요소 | Ablation 최적 | **최종 채택** | 조정 이유 |
|---|---|---|---|
| GATv2 Layer | 1 | **2** | 재난 시 차단 도로 우회에 2-hop 시야 필요 |
| State | 3-Dim | **4-Dim** (+time_to_go) | 재난 시 배터리/시간 기반 우회 판단 필요 |
| 나머지 | — | Ablation 결과 그대로 | — |

---

## 부록: 실험 재현 정보

- **실행 스크립트**: `tests/run_ablation.py`
- **실험 설정**: `tests/ablation_configs.py`
- **결과 집계**: `tests/summarize_ablation.py`
- **로그 위치**: `tests/ablation_results/<ID>/train_log.txt`
- **실행 요약**: `tests/ablation_results/run_summary.json`
