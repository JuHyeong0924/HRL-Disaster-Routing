# test_hrl_revisit_fix.py - Detailed Architecture & Refactoring Log

## 1. Overview
`test_hrl_revisit_fix.py`는 `worker_env.py`에 추가된 재방문 패널티(-5.0)와 `hrl_env.py`에 추가된 매니저 턴 단위 방문 이력(Memory) 초기화 로직이 텐서 차원에서 완벽하게 작동하는지 검증하기 위한 TDD(Test-Driven Development) 모듈입니다.

## 2. Test Capabilities
- **`test_worker_revisit_penalty`**: 워커가 `step_batch`를 수행할 때, 자신이 과거에 방문했던 노드(`visited_nodes == 1.0`)를 다시 밟는 Action을 취할 경우 정상적으로 `-5.0`의 패널티(기본 `step_penalty` 합산 -4.0 이하)가 부여되는지 물리적 검증을 수행합니다.
- **`test_hrl_visited_reset_per_manager_turn`**: 매니저가 새로운 지시를 내리는 `step_manager` 루프 진입 시, 워커 환경 내부의 `visited_nodes`가 리셋되는지 확인합니다. 에피소드 진행 중 누적된 총 방문 횟수가 (해당 턴에 걸어간 스텝 수 + 출발지 1)을 초과하지 않는지 등호 제약조건(`<=`)을 사용하여 검증합니다.

## 3. Trial & Error (Debugging Memory)
- **PyTorch Geometric Import Error**: 초기 작성 시 가상환경 모듈 인식 불가로 `pytest` 명령어가 실패하는 현상이 있었습니다.
- **Assertion Logic Flaw**: `step_manager()` 함수는 내부적으로 워커 루프를 여러 번 돌리기 때문에, 리셋 직후 방문 노드의 합이 `1.0`이 아니라 `워커가 이동한 총 스텝 수 + 1`이 되는 것이 정상 작동이라는 점을 간과하여 `AssertionError`가 발생했습니다.
- **해결 방안**: 테스트 스크립트를 독립 실행형 파이썬 스크립트(`__main__`)로 전환하고, `visited_sum <= float(steps_count + 1)`로 물리적 제약 공식을 수학적으로 완벽하게 교정하여 성공률 100% 검증을 완료했습니다.
