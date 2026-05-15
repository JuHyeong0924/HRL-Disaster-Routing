# HRL-Disaster-Routing: 상위 레벨 명세서 (High-Level Design)

본 문서는 프로젝트의 **전체 아키텍처, 모듈 간 관계, 학습 파이프라인 흐름**을 기술합니다.
함수 시그니처, 텐서 형태 등 구현 상세는 `project_specification.md`(하위 레벨 명세서)를 참조하세요.

---

## 1. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                   train_rl.py (진입점)                         │
│  --stage worker | manager_v2 | manager | alignment | phase1  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [worker]           [manager_v2]         [manager]  LEGACY   │
│  HRLWorkerTrainer   ManagerPPOTrainer    ManagerStageTrainer │
│  + HRLZoneEnv       + HRLClosedLoopEnv   + DisasterEnv      │
│  + Worker(4D)       + ReactiveManager    + NodeManager       │
│                     + Worker(4D-동결)    + Worker(7D-동결)    │
│                                                              │
│  [alignment] LEGACY                                          │
│  DOMOTrainer + DisasterEnv + NodeManager + Worker(7D-동결)   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.1. 계층형 강화학습(HRL) 구조

| 계층 | 역할 | 관찰 공간 | 행동 공간 |
|------|------|-----------|-----------|
| **Manager** | 거시적 경로 계획 (서브골 선택) | Node Graph (416노드, 4-Dim: is_curr, is_tgt, hop_dist, degree) | K-hop 반경 내 서브골 노드 선택 |
| **Worker** | 미시적 길찾기 (노드 단위) | Node Graph (416노드, 4-Dim: is_curr, is_tgt, is_next_zone, hop_dist) | 다음 노드 선택 |

### 1.2. Phase 구분

| Phase | 환경 | 재난 | Manager | Worker | 목표 |
|-------|------|:----:|---------|--------|------|
| **Phase 1** | `HRLZoneEnv` | ❌ | A* (Dummy) | RL 학습 | Zone 가이드 따라 길찾기 검증 |
| **Phase 2** | `HRLClosedLoopEnv` | ❌ | ReactiveManager PPO 학습 | 동결 (Phase 1 Best) | Closed-loop Re-planning 서브골 선택 |
| **Phase 3** | 미정 | ✅ | Fine-tune | Fine-tune | Manager-Worker 정렬 + 재난 대응 (향후 계획) |

---

## 2. 프로젝트 디렉토리 구조

```
HRL-Disaster-Routing/
├── train_rl.py                              # RL 학습 통합 진입점
├── train_sl.py                              # SL 사전학습 (레거시, 7-Dim용)
├── train.py                                 # 레거시 학습 스크립트
├── project_specification_hld.md             # 상위 레벨 명세서 (본 문서)
├── project_specification.md                 # 하위 레벨 명세서 (Code Map)
├── data/
│   ├── Anaheim_node.tntp                    # 노드 좌표
│   ├── Anaheim_net.tntp                     # 간선(도로) 데이터
│   ├── node_to_zone_k30.json               # METIS 분할 결과 (노드→Zone)
│   ├── zone_graph_k30.json                  # Zone 인접 그래프
│   └── hop_matrix_*.npy                     # APSP 홉 거리 캐시
├── src/
│   ├── envs/
│   │   ├── hrl_env.py                       # HRL Phase 1 환경 (배치 지원)
│   │   ├── hrl_closed_loop_env.py           # Manager-Worker Closed-loop 환경 (Phase 2)
│   │   ├── disaster_env.py                  # 재난 시뮬레이션 환경 (POMO, 레거시)
│   │   └── disaster_map.py                  # 물리 맵 엔진
│   ├── models/
│   │   ├── worker.py                        # Worker (4-Dim GATv2, Phase 1~2 공용)
│   │   ├── reactive_manager.py              # Manager v2 (GATv2, 비자기회귀 서브골 선택)
│   │   └── legacy/
│   │       ├── node_manager.py              # Node-level Manager (Transformer, 레거시)
│   │       └── zone_manager.py              # Zone-level Manager (GCN, 레거시)
│   ├── trainers/
│   │   ├── worker_trainer.py                # HRL Worker Trainer (Grad Accum, Phase 1)
│   │   ├── manager_ppo_trainer.py           # Manager v2 PPO Trainer (Phase 2)
│   │   └── legacy/
│   │       ├── worker_nav_trainer.py        # 레거시 Worker Trainer (7-Dim)
│   │       ├── manager_stage_trainer.py     # Manager 단독 RL 학습 (레거시)
│   │       └── pomo_trainer.py              # DOMOTrainer (Joint, 레거시)
│   ├── agents/
│   │   └── robot.py                         # BaseRobot/UGV 물리 에이전트 (향후 통합)
│   ├── data/
│   │   ├── generate_expert.py               # 전문가 경로 데이터 생성
│   │   └── segment_loader.py                # 세그먼트 데이터 로더
│   └── utils/
│       ├── graph_loader.py                  # TNTP → PyG 변환
│       ├── graph_converter.py               # 그래프 변환 유틸
│       └── types.py                         # Task/AgentState 데이터클래스 (향후 통합)
├── scripts/                                 # 유틸리티 스크립트
├── tests/                                   # 테스트 및 Ablation 설정
├── tools/                                   # 분석 도구
└── logs/                                    # 학습 로그 및 체크포인트
```

---

## 3. 데이터 흐름 개요

### 3.1. Phase 1 학습 루프 (Worker)
```
train_rl.py --stage worker
    │
    ├─ HRLZoneEnv.reset(batch_size=K)
    │   └─ A* 알고리즘 → Zone Sequence 생성
    │   └─ 4-Dim State [B, N, 4] 반환
    │
    ├─ Worker.forward(state, edge_index, mask)
    │   └─ GATv2 → Masked Softmax → Action 선택
    │
    ├─ HRLZoneEnv.step_batch(action)
    │   └─ Sliding Window 업데이트
    │   └─ Reward 계산 (Goal/Step/Invalid)
    │
    └─ Gradient Accumulation (K 에피소드 평균)
        └─ REINFORCE w/ Baseline 업데이트
```

### 3.2. Phase 2 학습 루프 (Manager v2)
```
train_rl.py --stage manager_v2
    │
    ├─ HRLClosedLoopEnv.reset()
    │   └─ 랜덤 OD쌍 선택 (current_idx, goal_idx)
    │
    ├─ ReactiveManager.select_action(state, edge_index, mask)
    │   └─ GATv2 → K-hop Masked Softmax → 서브골 1개 선택
    │
    ├─ HRLClosedLoopEnv.step(subgoal_idx)
    │   ├─ Worker.forward() (Greedy, 동결) × c_max 스텝
    │   └─ PBRS 보상: Φ(end) - Φ(start) - 0.1×steps
    │
    └─ ManagerPPOTrainer.update()
        ├─ RolloutBuffer → GAE(λ) Advantage 계산
        └─ PPO Clipped Objective + Critic MSE 업데이트
```

### 3.3. Zone 분할 (METIS)
```
원본 맵 (416 노드, 914 간선)
    │  METIS k-way partitioning
    ▼
K=30 Zone (구역당 ~14 노드)
    │  Zone 인접 그래프 생성
    ▼
Zone Graph (30 노드, ~60 간선)
```

---

## 4. 핵심 설계 결정 사항

| 항목 | 결정 | 근거 |
|------|------|------|
| Worker 상태 차원 | 4-Dim | Ablation 결과: 7-Dim과 성능 동등, 연산 50% 절감 |
| Worker Scorer | 2-Layer MLP | 단일 Linear 대비 표현력 향상 |
| Manager 아키텍처 | ReactiveManager (GATv2, 비자기회귀) | Transformer 자기회귀 대비 연산량 감소, K-hop 마스킹과 자연스러운 호환 |
| Manager 학습 | PPO + GAE (순수 RL) | SL Pretrain 제거로 파이프라인 단순화, Closed-loop 보상 활용 |
| Manager-Worker 연동 | Closed-loop Re-planning (PBRS) | POMO Joint 대비 안정적 학습, 순차적 모듈 분리 |
| Zone 분할 | METIS K=30 | 구역 간 Edge-cut 최소화, 균일 크기 보장 |
| Masking 방식 | Hard Masking (Phase 1) | 정적 맵에서는 구역 이탈 원천 차단이 안전 |
| 학습 전략 | Gradient Accumulation | GATv2 VRAM 제약으로 POMO 동시 Forward 불가 |
| Hardware | CPU 8 Threads, TF32, cuDNN Benchmark | 컨텍스트 스위칭 최소화 + GPU 커널 최적화 |
