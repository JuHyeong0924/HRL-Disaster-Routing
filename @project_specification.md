# HRL-Disaster-Routing: 상위 레벨 명세서 (High-Level Design)

본 문서는 프로젝트의 **전체 아키텍처, 모듈 간 관계, 학습 파이프라인 흐름**을 기술합니다.
함수 시그니처, 텐서 형태 등 구현 상세는 `project_specification.md`(하위 레벨 명세서)를 참조하세요.

---

## 1. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                   train_rl.py (진입점)                     │
│  --stage worker | manager | alignment | phase1           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  [worker]         [manager]          [alignment]         │
│  HRLWorkerTrainer ManagerStageTrainer DOMOTrainer        │
│  + HRLZoneEnv     + DisasterEnv       + DisasterEnv      │
│  + Worker(4D)     + NodeManager       + NodeManager      │
│                   + Worker(7D-동결)   + Worker(7D-동결)   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 1.1. 계층형 강화학습(HRL) 구조

| 계층 | 역할 | 관찰 공간 | 행동 공간 |
|------|------|-----------|-----------|
| **Manager** | 거시적 경로 계획 (Zone 단위) | Zone Graph (K개 노드) | 다음 Zone 선택 |
| **Worker** | 미시적 길찾기 (노드 단위) | Node Graph (416개 노드, 4-Dim) | 다음 노드 선택 |

### 1.2. Phase 구분

| Phase | 환경 | 재난 | Manager | Worker | 목표 |
|-------|------|:----:|---------|--------|------|
| **Phase 1** | `HRLZoneEnv` | ❌ | A* (Dummy) | RL 학습 | Zone 가이드 따라 길찾기 검증 |
| **Phase 2** | `DisasterEnv` | ✅ | GCN RL 학습 | 동결 | 재난 동적 우회 경로 학습 |
| **Phase 3** | `DisasterEnv` | ✅ | Fine-tune | Fine-tune | Manager-Worker 정렬(Joint) |

---

## 2. 프로젝트 디렉토리 구조

```
HRL-Disaster-Routing/
├── train_rl.py                          # RL 학습 통합 진입점
├── train_sl.py                          # SL 사전학습 (레거시, 7-Dim용)
├── data/
│   ├── Anaheim_node.tntp               # 노드 좌표
│   ├── Anaheim_net.tntp                # 간선(도로) 데이터
│   ├── node_to_zone_k30.json           # METIS 분할 결과 (노드→Zone)
│   └── zone_graph_k30.json             # Zone 인접 그래프
├── src/
│   ├── envs/
│   │   ├── hrl_env.py                  # HRL Phase 1 환경 (배치 지원)
│   │   ├── disaster_env.py             # 재난 시뮬레이션 환경 (POMO)
│   │   └── disaster_map.py             # 물리 맵 엔진
│   ├── models/
│   │   ├── worker.py                   # Worker (4-Dim GATv2, Phase 1 전용)
│   │   ├── node_manager.py             # Node-level Manager (Transformer)
│   │   └── zone_manager.py             # Zone-level Manager (GCN, Phase 2)
│   ├── trainers/
│   │   ├── worker_trainer.py           # HRL Worker Trainer (Grad Accum)
│   │   ├── worker_nav_trainer.py       # 레거시 Worker Trainer (7-Dim)
│   │   ├── manager_stage_trainer.py    # Manager 단독 학습
│   │   └── pomo_trainer.py             # DOMOTrainer (Joint 학습)
│   └── utils/
│       ├── graph_loader.py             # TNTP → PyG 변환
│       └── graph_converter.py          # 그래프 변환 유틸
└── logs/                                # 학습 로그 및 체크포인트
```

---

## 3. 데이터 흐름 개요

### 3.1. Phase 1 학습 루프
```
train_rl.py --stage worker
    │
    ├─ HRLZoneEnv.reset(batch_size=1)
    │   └─ A* 알고리즘 → Zone Sequence 생성
    │   └─ 4-Dim State [N, 4] 반환
    │
    ├─ Worker.forward(state, edge_index, mask)
    │   └─ GATv2 → Masked Softmax → Action 선택
    │
    ├─ HRLZoneEnv.step_batch(action)
    │   └─ Sliding Window 업데이트
    │   └─ Reward 계산 (Goal/Step/Invalid)
    │
    └─ Gradient Accumulation (K=16 에피소드 누적)
        └─ REINFORCE w/ Baseline 업데이트
```

### 3.2. Zone 분할 (METIS)
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
| Zone 분할 | METIS K=30 | 구역 간 Edge-cut 최소화, 균일 크기 보장 |
| Masking 방식 | Hard Masking (Phase 1) | 정적 맵에서는 구역 이탈 원천 차단이 안전 |
| 학습 전략 | Gradient Accumulation | GATv2 VRAM 제약으로 POMO 동시 Forward 불가 |
| SL 파이프라인 | 보존 (미사용) | Phase 2 Warm-start 가능성 대비 |
