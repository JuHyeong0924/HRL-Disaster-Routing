# 📝 Test HRL Env Aftershock Log (`tests/test_hrl_env_aftershock.py`)

## 1. 개요
*   **역할**: `src/envs/hrl_env.py`에 수정 적용된 시간 기반의 여진(Continuous Aftershock) 스케줄링 메커니즘을 실제 데이터 환경(Anaheim 도로 네트워크) 하에서 검증하기 위한 신규 단위 테스트 스크립트입니다.
*   **주요 검증 기능**:
    1.  여진 발생 총 횟수가 강화된 설계 기준 범위인 `15 ~ 25`회 내에 정상적으로 드로우되는지 여부.
    2.  생성된 여진 타임스탬프(`aftershock_times`) 리스트가 시간 정렬성을 유지하여 오름차순으로 배치되는지 여부.
    3.  모든 여진 시점이 유효 시각 범위(`[0, max_time]`) 내에 들어오는지 여부.

## 2. 세부 구현
- `pytest` 프레임워크를 기반으로 작성되었습니다.
- `MockWorker`나 더미 환경 대신, `WorkerEnv`에 직접 실제 데이터(`data/Anaheim_node.tntp`, `data/Anaheim_net.tntp` 및 METIS 구역 JSON 정보)를 투입해 연동 테스트를 진행합니다.
- 시드(seed) 변동성에 따른 예외 상황을 방지하기 위해 5회의 루프를 돌며 매 회차마다 `random`, `numpy`, `torch` 시드를 다양화하여 강건성을 확인합니다.

## 3. 검증 결과 (2026-06-27)
- `PYTHONPATH="" /home/sem/miniconda3/envs/rl/bin/python -m pytest tests/test_hrl_env_aftershock.py -v` 명령을 통해 로컬 검증 수행.
- **결과**: `1 passed`로 성공적으로 모든 요구 조건 통과.
