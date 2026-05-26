# HRL-Disaster-Routing: 상위 레벨 명세서 (High-Level Design)

본 문서는 프로젝트의 **전체 아키텍처, 모듈 간 관계, 학습 파이프라인 흐름**을 기술합니다.
함수 시그니처, 텐서 형태 등 구현 상세는 `@implementation_lld.md`(하위 레벨 명세서)를 참조하세요.

---

## 1. 시스템 아키텍처

```
┌────────────────────────────────────────────────────────────────────────┐
│                        train_rl.py (진입점)                            │
│                   --stage worker | manager_v2                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [worker]            [manager_v2]                                      │
│  HRLWorkerTrainer    ManagerPPOTrainer                                │
│  + HRLZoneEnv        + HRLClosedLoopEnv                               │
│  + Worker(4D)        + ReactiveManager                                │
│                      + Worker(4D-동결)                                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.1. 계층형 강화학습(HRL) 구조

| 계층 | 역할 | 관찰 공간 (State) | 행동 공간 (Action Space) |
|------|------|-----------|-----------| 
| **Manager** | 거시적 경로 계획 (서브골 선택) | Node Graph (416노드, 4-Dim: `[is_curr, is_tgt, hop_dist, degree]`) | K-hop 반경 내 서브골 노드 선택 (목적지 도달 시 direct 슛팅 예외 지원) |
| **Worker** | 미시적 길찾기 (노드 단위) | Node Graph (416노드, 4-Dim: `[is_curr, is_tgt, is_subgoal/next_zone, hop_dist]`) | 물리적 인웃 노드 선택 (`masking_mode` 및 `subgoal_mode` 분기 지원) |

*   **Subgoal Mode 분기:**
    *   **Zone 모드:** 서브골 영역(Zone) 소속 노드들을 활성화하여 가이드라인 제공.
    *   **Node 모드:** 3-hop 또는 6-hop 앞의 특정 물리적 서브골 노드를 활성화하여 다이렉트 가이드 제공.
*   **Masking Mode 분기 (Worker):**
    *   `hard`: {현재 Zone, 다음 Zone} 이웃만 허용 (구역 이탈 방지).
    *   `hard_full_seq`: Zone Sequence 전체에 속한 이웃 노드 허용.
    *   `soft_curr_next` / `soft_flex`: 물리적 제약 없이 모든 인접 노드를 허용하되, 구역 이탈 시 OOB 페널티(`-0.5`)로 간접 유도.

### 1.2. Phase 구분

| Phase | 환경 | 재난 | Manager | Worker | 목표 |
|-------|------|:----:|---------|--------|------|
| **Phase 1** | `HRLZoneEnv` | ❌ | A* (Dummy) / Node Sequence | RL 학습 (CosineLR, GAE, Entropy 옵션) | Zone 또는 Node 가이드에 따른 완벽한 길찾기 검증 |
| **Phase 2** | `HRLClosedLoopEnv` | ❌ | ReactiveManager PPO 학습 | 동결 (Phase 1 Best) | Closed-loop Re-planning에 의한 유연한 서브골 선택 최적화 |
| **Phase 3** | `DisasterEnv` (예정) | ✅ | Fine-tune | Fine-tune | Manager-Worker 정렬 및 동적 재난 환경 대응 (향후 계획) |

---

## 2. 프로젝트 디렉토리 구조

```
HRL-Disaster-Routing/
├── train_rl.py                              # RL 학습 통합 진입점 (--stage worker | manager_v2)
├── @architecture_hld.md                     # 상위 레벨 명세서 (본 문서)
├── @implementation_lld.md                   # 하위 레벨 명세서 (Code Map)
├── data/
│   ├── {Map}_node.tntp                      # 노드 좌표 (Anaheim, ChicagoSketch, Goldcoast, SiouxFalls)
│   ├── {Map}_net.tntp                       # 간선(도로) 데이터
│   ├── node_to_zone_{Map}_k*.json           # METIS 분할 결과 (노드 → Zone ID)
│   ├── zone_graph_{Map}_k*.json             # Zone 인접 그래프 메타데이터
│   ├── hop_matrix_*.npy                     # BFS 기반 APSP 홉 거리 자동 캐시 파일
│   └── cache/                               # APSP 캐시 (DisasterEnv용, Phase 3 대비)
├── src/
│   ├── envs/
│   │   ├── hrl_env.py                       # HRL Phase 1 환경 (POMO 배치 병렬 및 masking_mode/subgoal_mode 내장)
│   │   ├── hrl_closed_loop_env.py           # Manager-Worker Closed-loop 상호작용 환경 (Phase 2, PBRS 및 엣지 정규화 지원)
│   │   ├── disaster_env.py                  # 재난 시뮬레이션 환경 (Phase 3 대비 보존)
│   │   └── disaster_map.py                  # SiouxFalls/Anaheim TNTP 물리 맵 엔진 (HAZUS 5단계 파괴도 지원)
│   ├── models/
│   │   ├── worker.py                        # Worker (GATv2, GraphNorm, Residual, Gradient Checkpoint, JK-Net)
│   │   └── reactive_manager.py              # Manager v2 (GATv2, Dual Head Actor-Critic, 비자기회귀 서브골 Scorer)
│   ├── trainers/
│   │   ├── worker_trainer.py                # Worker Trainer (POMO 병렬화, GAE, Entropy, Cosine LR 지원)
│   │   └── manager_ppo_trainer.py           # Manager v2 PPO Trainer (GAE, Advantage 정규화, learning_curve 자동 시각화)
│   ├── agents/
│   │   └── robot.py                         # BaseRobot/UGV (RoboCue-X 물리 주행 및 전고체 배터리 소모 스펙 탑재)
│   └── utils/
│       ├── graph_loader.py                  # TNTP → PyG 변환
│       ├── graph_converter.py               # 그래프 변환 유틸
│       └── types.py                         # Task(임무), AgentState(실시간 상태) Dataclass 명세
├── scripts/                                 # 유틸리티 스크립트
├── tests/                                   # 평가 및 Ablation 아카이브
│   ├── ablation_results/                    # Worker ablation 보고서 및 학습 로그 (v1~v5)
│   └── paper_figures/                       # 논문용 시각화 자료
├── tools/                                   # 분석 도구
└── logs/                                    # 학습 로그 및 체크포인트
    ├── rl_worker_stage/                     # Phase 1 Worker 체크포인트 (활성)
    └── rl_manager_v2/                       # Phase 2 Manager v2 체크포인트 (활성)
```

---

## 3. 데이터 흐름 개요

### 3.1. Phase 1 학습 루프 (Worker)
```
train_rl.py --stage worker
    │
    ├─ HRLZoneEnv.reset(batch_size=K)
    │   ├─ 무작위 시종착점 선택 (서로 다른 Zone)
    │   ├─ A* (Zone) 또는 최단 경로 (Node) 시퀀스 빌드
    │   └─ [B, N, 4] 텐서 형태로 State 초기화 (max_hop 기반 정규화 적용)
    │
    ├─ Worker.forward(x_flat, aei, batch=ai, neighbors_mask=mask_flat)
    │   ├─ Gradient Checkpointing 적용 (VRAM 절약)
    │   ├─ GATv2Conv 공간 인코딩 + GraphNorm + Residual + JK-Net 결합
    │   └─ Softmax → Categorical 행동 샘플링 (probs_all [B*N], values_all [B, 1])
    │
    ├─ HRLZoneEnv.step_batch(actions)
    │   ├─ 물리적 인접성 및 제자리(Stagnation) 검사
    │   ├─ masking_mode에 따른 Zone 위반 여부 판정 (soft일 경우 OOB 페널티 부여)
    │   ├─ Sliding Window 업데이트 (목표 Zone/Node 도착 시 인덱스 증분)
    │   └─ [use_pbrs] 홉 거리 포텐셜 차이 기반 Dense Reward 추가
    │
    └─ Gradient Accumulation (K 에피소드 평균 합산 역전파)
        └─ GAE(λ) 또는 Monte Carlo 기반 Advantage 산출 → Optimizer 업데이트 (CosineLR 지원)
```

### 3.2. Phase 2 학습 루프 (Manager v2)
```
train_rl.py --stage manager_v2
    │
    ├─ HRLClosedLoopEnv.reset()
    │   └─ 랜덤 도달 가능 OD쌍 선택 (current_idx, goal_idx)
    │
    ├─ ReactiveManager.select_action(x, edge_index, current_idx, goal_idx, candidate_mask)
    │   ├─ GATv2 공간 인코딩 + GraphNorm
    │   ├─ Actor Head: h_curr ∥ h_goal ∥ h_candidate → MLP → Score
    │   ├─ Critic Head: h_curr ∥ h_goal → MLP → V(s) 가치 추정
    │   └─ K-hop 마스킹 (목적지 범위 내 진입 시 예외 허용) → Categorical 샘플링
    │
    ├─ HRLClosedLoopEnv.step(subgoal_idx)
    │   ├─ Worker.forward() (Greedy, 동결 상태) 호출하여 서브골을 향해 최대 c_max 스텝 전진
    │   └─ PBRS dense 보상: Φ(end) - Φ(start) - (step_penalty * steps) + goal_bonus
    │
    └─ ManagerPPOTrainer.update()
        ├─ RolloutBuffer 경험 수집 → compute_gae(gamma, lam)
        ├─ Advantage Z-score 정규화 ((adv - mean) / (std + 1e-8))로 학습 안정성 극대화
        └─ PPO Clipped Objective + Critic MSE 손실 합산 최적화 (n_epochs 반복)
```

---

## 4. 핵심 설계 결정 사항

| 항목 | 결정 | 근거 |
|------|------|------|
| **Worker 상태 차원** | 4-Dim | Ablation 결과: 7-Dim 대비 물리적 성능 차이가 없으면서 연산 효율 50% 절감 |
| **GNN 최적화 플래그** | Checkpoint, JK-Net, Edge Features | GATv2 레이어가 깊어질 때의 VRAM 극복 및 그래프 위상학적 특성 정보 전달력 증대 |
| **Manager 아키텍처** | ReactiveManager (비자기회귀) | Transformer 자기회귀 디코더 대비 파라미터 수 및 연산 부하 대폭 절감, K-hop 물리 영역 바인딩 |
| **Advantage Z-score 정규화** | Z-score Normalization | PPO 업데이트 단계에서 Advantage의 scale 편차를 제거하여 Gradient Exploding 방지 및 고속 수렴 달성 |
| **HAZUS 5단계 물리 모델** | 험지 및 데미지 부하 계수 | 단순 거리가 아닌 도로의 붕괴 및 물리적 상태를 배터리 소모 전비(Wh/km)와 주행 속도에 정밀 투영 |

---

## 5. 참고 문헌 (Literature References)

최신(2024~2026) arXiv 연구 동향 분석(`hrl_research_report_v2.md`)에 따라, 본 아키텍처는 다음 연구들과 유사한 궤를 갖습니다:
1. **Congestion Estimation 기반 Safe Mapless Navigation (arXiv:2503.12036):** 혼잡도 평가를 결합하여 Local Minima를 회피하는 서브골 생성. (본 프로젝트의 Manager 혼잡도/데미지 반영 로직과 일치)
2. **CBF(Control Barrier Functions) 결합 HRL (arXiv:2501.17424):** 하위 레벨에 안전 필터를 결합한 계층적 강화학습 구조. (향후 Worker의 충돌 회피/안전 제약에 적용 가능)
3. **해마-선조체 영감 High-reward Graph Planning (arXiv:2410.09505):** 고수익(High-reward) 노드 위주의 메모리 그래프 구축. (본 프로젝트의 Node Graph 기반 서브골 선택과 매우 유사)
