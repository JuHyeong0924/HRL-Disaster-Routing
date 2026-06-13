# Tier 1: System Context (Macro Overview)

## 1. Project Purpose
HRL-Disaster-Routing 프로젝트는 재난 상황과 같은 복잡하고 동적인 도로망 환경에서 계층적 강화학습(Hierarchical Reinforcement Learning, HRL)을 통해 최적의 대피 경로 또는 구호 물류 라우팅을 수행하는 시스템입니다. 단일 에이전트가 복잡한 그래프의 모든 노드를 탐색하는 기존 Flat RL 방식의 한계(차원의 저주, 탐색의 비효율성)를 극복하기 위해, 전체 그래프를 여러 Zone으로 분할(Partitioning)하고 Manager-Worker 구조로 역할을 나누어 탐색 효율과 범용성을 극대화하는 것을 목적으로 합니다.

## 2. Macro Topology & Core Paradigm
이 시스템은 크게 **그래프 파티셔닝(Graph Partitioning)**, **Manager (High-level Policy)**, **Worker (Low-level Policy)** 세 가지 핵심 도메인으로 나뉩니다.

### 2.1. Graph Partitioning (Zone 생성)
- **도구**: `src/utils/generate_zones.py` (METIS 기반)
- **역할**: 거대한 물리적 네트워크 그래프(Node/Edge)를 K개의 서브 그래프(Zone)로 군집화합니다.
- **특징**: 단순한 지리적 분할이 아닌 노드 간 연결성을 고려한 위상적 파티셔닝을 수행하여, Worker가 한 Zone 내부를 이동할 때 병목 현상 없이 효율적으로 탐색할 수 있도록 보장합니다.

### 2.2. HRL 구조: Manager와 Worker의 협력 (Closed-loop)
HRL 구조는 두 단계(Phase)의 학습 및 실행으로 구성됩니다.
1. **Worker (Phase 1)**: 특정 Zone 내에서 시작점으로부터 목적지 노드(또는 다른 인접 Zone의 경계 노드)까지 최단/최적 경로로 이동하는 방법을 학습합니다. GNN(GATv2 등)을 사용하여 Node와 Edge(거리, 속도, 수용량 등)의 특징을 이해하고 다음 방문할 노드를 선택합니다.
2. **Manager (Phase 2)**: 전체 맵의 Zone 그래프 단위에서 작동하며, 현재 Zone에서 목적지 Zone으로 가기 위해 다음으로 진입해야 할 중간 Subgoal Zone을 결정합니다. Manager는 비자기회귀(Non-autoregressive) 방식 또는 PPO(Proximal Policy Optimization)를 사용하여 거시적인 탐색 방향을 지시합니다.

- **실행 흐름 (Evaluation/Inference)**: 
  Manager가 `Subgoal Zone Z`를 지정 $\rightarrow$ Worker가 해당 Zone $Z$로 들어가기 위해 현재 위치에서 연결된 노드들을 따라 탐색 $\rightarrow$ Worker가 다음 Zone에 진입하거나 최대 턴(c_max)을 소모하면 다시 Manager에게 제어권이 넘어옴. 

## 3. Directory Structure (Architecture Map)
시스템의 물리적 디렉토리 구조는 다음과 같이 역할을 분담합니다.

- `src/envs/`: RL 학습을 위한 환경 모듈. Manager를 위한 환경(`manager_env.py`), Worker를 위한 환경(`worker_env.py`), 그리고 Zero-shot 테스트 및 HRL 통합 제어를 위한 환경(`hrl_env.py`)이 위치.
- `src/models/`: GNN 기반의 신경망 아키텍처. Manager 망(`manager.py`)과 Worker 망(`worker.py`).
- `src/trainers/`: PPO 알고리즘 기반 학습 루프 구현체 (`manager_trainer.py`, `worker_trainer.py`).
- `src/utils/`: 데이터 로드, 타입 정의, Zone 그래프 파티셔닝 등 유틸리티 스크립트.
- `scripts/`: 모델 학습 진입점. 주로 `train_rl.py`가 배치.
- `tests/`: 훈련된 모델의 성능 평가 모듈. HRL/FlatRL 비교 및 Cross-map Zero-shot 평가를 수행하는 `evaluate.py`.
- `docs/`: 3-Tier 시스템 설계 가이드 및 로그.
- `data/`: TNTP 형식의 도로망 원본 데이터와 파티셔닝된 Zone JSON 데이터가 저장됨.

## 4. Key Capabilities & Features
- **Cross-map Zero-shot Transfer**: 특정 맵(예: Anaheim)에서 학습된 Worker(지역 이동 정책)가 다른 맵(예: Chicago, Berlin)에서도 재학습 없이 동작하도록 설계. 이는 Manager가 지시하는 Subgoal Zone과 Worker의 Local Feature 중심 GNN 모델 구조 덕분.
- **Dynamic Masking & PBRS**: Action Masking (Soft/Hard) 기법과 PBRS (Potential-Based Reward Shaping)를 도입하여 에이전트의 불필요한 Cyclic 경로를 제한하고 빠른 목표 도달 유도.
- **Ablation Ready**: `baseline` (Flat RL), `use_global_pool`, `use_is_visited` 등 다양한 구조적 변형을 플래그를 통해 손쉽게 실험할 수 있도록 모듈화.
