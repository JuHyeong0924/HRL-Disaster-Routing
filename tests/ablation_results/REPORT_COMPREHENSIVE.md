# HRL Worker Ablation Study: 종합 보고서 (v1 ~ v4)

본 보고서는 HRL(Hierarchical Reinforcement Learning) 프레임워크 내 최하위 노드 탐색을 담당하는 **Worker 에이전트**의 최적화를 위해 진행된 네 차례의 Ablation Study(v1 ~ v4) 결과를 총망라한 종합 분석 보고서입니다.

---

## 1. v1: State & Architecture (레거시 7-Dim Pipeline)
**"불필요한 정보를 덜어내고 핵심 구조를 확립하다"**

초기 Worker 모델(`WorkerLSTM`)은 7차원(7-Dim)의 상태 벡터와 LSTM 기반의 시계열 처리, 그리고 복잡한 보상 체계를 가졌습니다. v1 실험은 이 무거운 구조를 경량화하고 최적화하는 데 집중했습니다.

- **상태 표현 (State) 최적화**: 오직 **3-Dim (`is_curr`, `is_tgt`, `hop_dist`)** 핵심 정보만 남겼을 때 99.9%의 성공률을 달성.
- **시계열 아키텍처**: LSTM 모듈을 제거하고 **단순 Linear 투영**으로 변경했을 때 성공률이 10% 이상 상승.
- **보상 구조**: **Goal 도달 보상과 Step 페널티**만 남긴 간소화된 보상 체계가 가장 강력한 수렴(99.7%).
- **결론**: **가벼운 State + 간결한 Reward + LSTM 제거 + Residual/GraphNorm 유지**라는 Worker의 기본 뼈대를 확립.

| 카테고리 | Best ID | 설명 | EMA |
|---|---|---|---|
| Architecture | A5 | LSTM 제거 (Linear 투영) | 91.4% |
| State | S5 | 최소 State (3-Dim) | 99.9% |
| Reward | R5 | 간소화 보상 (goal+step만) | 99.7% |

---

## 2. v2: RL Algorithm 안정화 (4-Dim HRL Pipeline)
**"RL 알고리즘의 안정화와 Dense Reward 환경 구축"**

v1의 결론을 바탕으로 `HRLZoneEnv`라는 Zone 기반 환경으로 전환한 뒤, REINFORCE 알고리즘의 학습 안정성과 탐색 최적화를 위해 진행.

- **Zone Progress Reward**: Zone 전환 시 중간 보상을 부여하여 학습 초기의 기울기 소실을 방지.
- **GAE(Generalized Advantage Estimation)**: 분산을 줄여 학습 안정성을 극적으로 향상.
- **Entropy Regularization (0.01)**: 조기 수렴 방지 및 충분한 상태 탐색 유도.
- **Cosine LR Schedule**: 후반부 미세 조정을 통한 최종 수렴 품질 향상.
- **결론**: **GAE + Entropy + Zone Reward**를 기본 파이프라인으로 채택하여 100% 수렴을 위한 알고리즘적 토대 완성.

---

## 3. v3: Masking & PBRS (해의 품질 개선)
**"성공률 100%의 함정을 넘어 최적 경로(Optimal Path)를 찾다"**

모델들이 대부분 100% 성공률(SR)에 도달하는 포화 현상을 극복하고, 경로 효율성(Path Length)을 평가하는 데 집중.

- **`soft_curr_next`**: 유효 노드를 타겟 주변으로 극단적으로 축소하여 **가장 짧은 최적 경로(Len=15.5)**를 찾도록 유도.
- **PBRS 시너지**: 목표를 향해 다가갈수록 부드러운 그래디언트를 제공, 평균 보상(Rw)이 +6~7점 상승.
- **결론**: **`soft_curr_next` + PBRS + 3-Layer GATv2** 구조를 HRL Worker의 최적 형태로 확정.

| ID | 설정 | SR | 평균 경로 길이 |
|---|---|---|---|
| **SCN_P** | `soft_curr_next`, PBRS O | 100.0% | **15.5** |
| P_SF | `soft_flex`, PBRS O | 100.0% | 18.6 |
| M_HFS | `hard_full_seq` | 78.0% | - |

---

## 4. v4: Edge-Conditioning & JK-Net (아키텍처 혁신)
**"Worker의 완벽한 아키텍처를 완성하다"**

- **Edge-Conditioning**: 엣지 피처(거리/용량/속도)를 노드 간 어텐션 계산에 포함시켜, 병목이나 긴 도로를 회피하는 능력을 크게 향상.
- **Jumping Knowledge (JK-Net)**: Over-smoothing 현상을 해결하기 위해 각 레이어의 출력을 `cat`으로 합쳐 성능 극대화.
- **결론**: `Len=12.6` 홉, 성공률 100%라는 최고 수준의 Worker 기본 주행 성능 완성.

| ID | 설명 | SR | 평균 경로 길이 |
|---|---|---|---|
| **JK3_EC** | 3-Layer + JK-Net + Edge-Conditioning | 100.0% | **12.6** |
| EC3 | 3-Layer + Edge-Conditioning | 100.0% | 13.8 |
| BL_L3 | 3-Layer Baseline | 100.0% | 15.5 |

---

## 종합 요약: 최종 확정 Worker 아키텍처

| 항목 | 확정 값 | 근거 |
|---|---|---|
| **State Dim** | 4-Dim | v1: S5(3-Dim)=99.9% |
| **GNN** | 3-Layer GATv2 + Residual + GraphNorm | v1: A5, v3: L3 |
| **Temporal** | Linear 투영 (LSTM 제거) | v1: A5=91.4% vs BASELINE=81.4% |
| **Masking** | `soft_curr_next` | v3: Len=15.5 (최단) |
| **Reward** | Goal + Step + PBRS | v1: R5=99.7%, v3: +6.0 Rw |
| **Edge Features** | Edge-Conditioning | v4: JK3_EC=Len 12.6 |
| **JK-Net** | Jumping Knowledge (`cat` 모드) | v4: Over-smoothing 방지 |
| **RL 알고리즘** | REINFORCE + GAE(λ=0.95) + Entropy(0.01) + Cosine LR | v2: P0P1P2=100% |

### 핵심 원리

1. **차원 축소의 위력**: v1(7→3차원), v3(`soft_flex`→`soft_curr_next`) — 에이전트가 신경 써야 할 피처의 수를 줄일수록 RL은 더 짧고 완벽한 해를 찾아냄.
2. **보상 함수의 진화**: 복잡한 휴리스틱(v1) → 단계적 보상(v2) → 수학적 PBRS(v3) — Policy Gradient의 분산을 단계적으로 통제.
3. **아키텍처 혁신**: v4에서 JK-Net + Edge-Conditioning을 통해 15.5→12.6 홉 최적 경로 도출 성공.
