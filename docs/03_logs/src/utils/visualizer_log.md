# `src/utils/visualizer.py` Log

## 1. 코드 상세 설명
- **역할**: Matplotlib 기반 물리적 도로망, 타겟, 워커 이동 시각화.
- **핵심 로직 변경 (Continuous Time Update)**:
  - `plot_state` 함수 서명을 변경하여, 0.5초 틱 기반의 `global_time`, 워커가 위치한 현재 간선 정보 `worker_edge`, 그리고 간선 위 이동 비율 `worker_progress`를 전달받아 시각화하도록 업그레이드하였습니다.
  - 핵심 로직 - Target 오버레이(Time Window): `ax.text` 대신 `ax.annotate`와 `textcoords='offset points'`, `xytext=(0, 15)`를 도입하여 맵의 스케일 크기(Anaheim, Berlin 등)에 구애받지 않고 항상 미션 타겟 아이콘의 정중앙 상단에 15픽셀 띄운 상태로 안정적으로 텍스트가 따라다니게 하였습니다.
  - Failure 시각화: 시간이 초과되어 `target_failed`가 True가 된 목표물은 지도 위에서 렌더링을 완전히 생략(`continue`)시켜, 미션 목록에서 폐기(삭제)되었음을 직관적으로 알림.
  - 기존에는 단순히 특정 노드(`curr_node`)에 고정된 형태로 워커를 점프시키듯 그렸으나, 이제는 선형 보간식을 도입해 워커 아이콘을 부드럽게 렌더링합니다.
  - UI 텍스트 상단의 "Time" 지표를 홉 스텝 수가 아닌 물리적 실수형 타이머(`global_time`)로 표시합니다.

## 2. 시각화 알고리즘
- 파괴된 간선의 선 두께를 `1.0`에서 `3.0`으로 증가시켜, 노란색~검붉은색(`YlOrRd`) 그라데이션 컬러맵 점선의 시인성을 대폭 강화.
- 워커 이동 애니메이션 보간식: `wx = ux + (vx - ux) * worker_progress` (y좌표도 동일). `worker_edge` 파라미터가 유효하면 간선 위의 점을 도출하여 렌더링하고, 유효하지 않은 경우 정수 노드 위에 워커를 렌더링합니다.

## 3. Trial & Error (Troubleshooting)
- **이슈 1**: 연속 시간(Continuous Time) 모델 도입에 따른 렌더링 부자연스러움 해소 필요.
  - **해결**: 워커의 이동률(Progress) 파라미터를 추가로 입력받아 Matplotlib의 X/Y 좌표를 동적으로 보간계산(Interpolate)함으로써 애니메이션이 뚝뚝 끊기지 않고 부드럽게 가시화되도록 고도화.
