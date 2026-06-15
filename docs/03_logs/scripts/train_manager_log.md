# `train_manager.py` Log

## Detailed Code Explanation
- **배치 및 미니 배치 파라미터 개편**:
  `--batch_size`의 기본값을 32에서 256으로 늘려 24GB VRAM 활용률을 극대화했습니다.
  `--mini_batch_size` 인자를 신규로 추가하여, 배치 사이즈가 늘어난 만큼 PPO Backward Update 횟수가 기형적으로 증가하는 것을 방지했습니다. (기본값 2048)
- `HRLEnv` 인스턴스 생성 시 `max_time`, `max_manager_turns`를 주입하던 레거시 코드를 삭제했습니다. (해당 값들은 이제 `hrl_env.py` 내부 `reset()`에서 동적으로 주입됨)
- `ManagerPPOTrainer` 에 `args` 객체 자체를 `config` 파라미터로 넘겨주어 `mini_batch_size`를 자동으로 파싱하도록 구조를 개선했습니다.

## Trial & Error Log
- VRAM 2GB 이슈 해결 및 미니배치 부재로 인한 PPO 오버피팅 가능성을 선제적으로 차단했습니다.
