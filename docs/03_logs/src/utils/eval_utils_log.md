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

## 치명적 버그 수정: Worker num_layers 하드코딩 해결 및 지표 개선 (2026-06-27)

### 변경 개요
*   평가 환경(`eval_utils.py`)에서 Worker 네트워크 인스턴스를 생성할 때 GNN 레이어 수(`num_layers=2`)가 고정되어 있어, 4-레이어로 학습된 체크포인트 가중치를 불러올 때 상위 GAT 레이어 가중치(Conv 2, 3 등 총 20개 Key)가 소실되던 치명적인 불일치 버그를 해결함.
*   에피소드 통계에서 단순 실패 플래그(`failed`)를 수집하던 방식에서, 실제 UGV 파괴 대수를 반영하도록 `ugv_destroys` 변수를 획득해 리턴하도록 평가 루프를 개선함.

### 핵심 변경 사항
1.  **`load_eval_env()` 함수 매개변수화**: GNN 레이어 수를 동적으로 지정하기 위해 `num_layers: int = 4` 매개변수 추가 및 더미 Worker 인스턴스화 시 적용. 기본 `disaster_prob`을 `0.2`로 기본값 격상.
2.  **`load_neural_models()` 함수 매개변수화**: 체크포인트 레이어 수와의 일치를 보장하기 위해 `num_layers: int = 4` 매개변수 추가 및 Worker 인스턴스화 시 적용.
3.  **`run_evaluation_episode()`의 UGV 파괴 횟수 반환**: 기존 boolean형 `failed` 반환 대신, 환경 인스턴스 `hrl_env.ugv_destroys[0].item()` 값을 추출하여 UGV 파괴 대수(`ugv_destroys`)를 직접 리턴하도록 개선.

### Trial & Error
*   **이유 규명**: 이전 벤치마크 테스트에서 Layer-4 Worker를 제대로 평가하였음에도 ALNS-Dijkstra 대비 현저히 낮은 성능(구출률 60%선)을 기록한 근본 원인이 이 하드코딩 버그 때문이었음. PyTorch가 `strict=False`로 가중치를 불러와 겉으로는 에러 없이 작동했으나, 실질적으로 모델의 약 38%에 달하는 가중치를 드롭한 채 평가되고 있었음.

