# `manager_ppo_trainer.py` Log

## Detailed Code Explanation
- **PBRS 정규화(Log1p Scaling) 도입**:
  기존에는 `dist_matrix`의 원시(Raw) 거리 값을 그대로 사용하여 `prev_dist - curr_dist`를 계산했습니다. 그러나 여진(`apply_aftershock`) 발생 시 특정 간선의 가중치가 50배 폭증함에 따라, 다익스트라 최단 거리가 수만 단위로 폭주하는 현상이 발견되었습니다. 이로 인해 단일 턴의 PBRS 보상이 수만 단위의 음수 페널티로 변질되어 가치 신경망의 Loss를 비정상적으로 발산시켰습니다.
  이를 해결하기 위해 `np.log1p()` 스케일링을 적용하여 거리를 로그 스케일로 압축했습니다.
  `reward_pbrs = (np.log1p(prev_dist) - np.log1p(curr_dist)) * 2.0`
  이 수식은 최대 거리가 80,000을 넘어가도 11.3 수준으로 억제하며, 거리가 단축될 때 주어지는 PBRS 보상의 총합이 실질적인 타겟 구출 보상(+10)과 부합하도록 스케일 계수(2.0)를 맞췄습니다.

## Trial & Error Log
- **오류 증상:** `Loss=3158`, `V=6316`, `Rw=-733`
- **해결 내역:** `np.log1p` 적용 후 `/validation` 테스트 결과, 극단적인 페널티 폭탄이 사라지고 모델의 가치 신경망 학습 안정성(Loss 수치 감소 및 정상화)이 회복되었습니다.
