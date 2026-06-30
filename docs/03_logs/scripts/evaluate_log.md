# 📝 Evaluate Log (`scripts/evaluate.py`)

## 1. 개요
*   **역할**: 학습된 신경망 모델(HRL, GA-Neural, ALNS-Neural) 및 전통적인 휴리스틱 알고리즘(Dijkstra, GA-Dijkstra, ALNS-Dijkstra)의 성능을 시뮬레이션 환경 내에서 비교 평가하는 다목적 평가 스크립트입니다.
*   **지원 모드**:
    1.  `benchmark`: 다수의 시나리오(에피소드) 하에서 구출률, 연산 지연시간, 경로 재계산 횟수, UGV 파괴 횟수를 종합 산출하는 성능표 출력 모드.
    2.  `analyze`: 매 스텝의 의사결정 상태를 정밀 로그로 터미널에 기록하는 모드.
    3.  `visualize`: 시뮬레이션 진행 상황을 GIF 이미지 프레임으로 렌더링 및 저장하는 모드.

## 2. 변경 로그 (2026-06-27)

### 2.1. 동적 재난 인자 추가 및 하드코딩 제거
- **argparse 인자 추가**: `--num_layers` (GNN 레이어 수, 기본값 4), `--disaster_prob` (재난 발생 확률, 기본값 0.2), `--dynamic_disaster`/`--no_dynamic_disaster` (동적 여진 활성화 여부) 인자를 추가하여 하드코딩된 환경 변수들을 런타임 매개변수화함.
- **최신 체크포인트 기본 경로 지정**: 6월 26일 자 최신 학습 체크포인트(`2026-06-26_151453_worker/best.pt` 및 `2026-06-26_190100_manager/best_manager.pt`)를 파서 기본값으로 업데이트.
- **파라미터 전달 연동**:
  - `load_neural_models()` 호출 시 `num_layers` 전달.
  - `load_eval_env()` 호출 시 `disaster_prob`, `dynamic_disaster`, `num_layers`를 동적으로 전달하도록 벤치마크 루프를 갱신함.

### 2.2. UGV 파괴 대수 정확성 확보
- 기존의 에피소드 단순 실패 카운트(`failures`) 대신, 에피소드 당 파괴된 UGV 대수의 실제 평균값을 얻도록 `results` 딕셔너리의 키를 `ugv_destroys` 리스트 형태로 개편.
- 테이블 헤더 출력을 `Fails`에서 `UGV Destroys`로 변경하고 `np.mean` 수치를 출력하도록 로직을 수정함.

## 3. Trial & Error
- **GA 솔버 무한 루프 현상**: 재난 강도를 0.2로 대폭 강화했을 때, GA 휴리스틱 계열 모델들이 연산 시간 초과(Timeout) 혹은 무한 루프 상태에 진입하여 전체 벤치마크 태스크가 정지되는 현상이 발견됨.
- **조치**: 환경 격리 벤치마크 테스트 진행 시에는 GA/ALNS-Neural 등 불필요한 비교군을 일시 차단하고, HRL 및 HRL-Dijkstra만 선별적으로 구동하는 파이썬 코드를 작성하여 효율적으로 격리 테스트를 마무리함.
