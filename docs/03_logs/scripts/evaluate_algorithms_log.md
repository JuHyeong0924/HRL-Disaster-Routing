# `scripts/evaluate_algorithms.py` 로그

## 상세 코드 설명
- **역할**: HRL 구조(Neural Manager + Neural Worker)와 베이스라인 휴리스틱 모델들을 동일한 시드와 환경 조건 하에서 공정하게 정량 평가(Benchmark)하는 실행 스크립트입니다.
- **주요 로직**:
  - `--episodes`, `--map`, `--num_targets` 인자를 받아 환경을 통제합니다.
  - 사전에 학습 완료된 최고 성능(`best.pt`)의 신경망 워커 및 매니저 가중치를 불러와(`torch.load`) 초기화합니다.
  - 5가지 조합(HRL, GA-Neural, ALNS-Neural, GA-Dijkstra, ALNS-Dijkstra)을 튜플 리스트(`models_to_test`)로 구성합니다.
  - 동일한 `seed`를 강제 부여하여 초기 지진(Static Disaster)과 타겟 좌표를 완벽히 일치시키고 `evaluate_model` 함수를 호출하여 성능을 측정합니다.
  - Rescue Rate(%), Latency(s), Recomputes, UGV Destroys 네 가지 핵심 지표를 표 형태로 표준 출력(`stdout`)합니다.
- **텐서 논리**:
  - `evaluate_model` 내에서 HRL Manager 호출 시, 현재 환경(`hrl_env`)에서 추출한 `zone_features[K, 6]`, `target_features[N, 4]` 등을 신경망 입력에 맞게 포매팅 및 배치화(`batch=ai`)하여 전달합니다.
  - 히든 스테이트 `h_last` 텐서는 에피소드 진행 중 메모리를 유지해야 하므로 `[1, 128]` 크기로 선언되어 타임 스텝마다 재사용/갱신됩니다.

## 시행착오 (Trial & Error)
- **에러 (메모리 손실 버그)**: 처음 구현 시 모든 알고리즘의 구조율이 0%로 측정되었습니다.
- **분석 (`debugging` 단계)**: `sequential_thinking`을 통해 디버깅한 결과, `h_last = torch.zeros(1, 128)` 선언부가 `while not done:` 루프 안쪽에 위치해 있어 매 스텝마다 HRL 매니저가 기억을 리셋당하는 치명적인 논리 오류를 발견했습니다. 또한 무작위 가중치(Random Init)된 모델을 평가하던 버그도 발견했습니다.
- **해결**: `h_last`를 루프 외부로 빼내어 순환 신경망/트랜스포머의 과거 상태 누적을 정상적으로 보장하고, `best_manager.pt` / `best.pt` 가중치를 로드하는 코드를 상단에 추가하여 정상적인 성능이 도출되도록 수정했습니다.
