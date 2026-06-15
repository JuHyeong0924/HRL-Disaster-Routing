# `hrl_env.py` Log

## Detailed Code Explanation
- **Dynamic Limits (동적 한계치 설정)**: 
  `__init__` 함수에 고정되어 있던 `max_time=200`, `max_manager_turns=50` 하드코딩을 제거했습니다.
  대신 `reset(num_targets)` 함수 내부에서 `self.max_manager_turns = num_targets * 20`, `self.max_time = num_targets * 80` 로 동적으로 계산하여 할당합니다.
  이는 타겟 개수가 늘어나거나 줄어들 때, 물리적으로 도달할 수 있는 턴과 시간 한계를 알아서 비례 확장(Scale)하여 "불가능한 미션"이 되는 것을 방지하는 목적입니다.

## Trial & Error Log
- 타겟을 찾기 전에 에피소드가 강제로 종료되는 문제(Sparse Reward 극대화 원인)를 이 동적 계산으로 해결했습니다.
