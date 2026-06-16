# State 02: System Design & Blueprint

## 1. Architectural Changes

### 1.1 `src/envs/worker_env.py`
**Goal:** Worker Policy가 "이미 방문했던 노드(Revisiting)"를 자발적으로 기피하는 정책을 학습하도록, 명확한 음의 보상(Negative Reward)을 부여합니다.
- `step_batch()` 함수 내부 수정:
  ```python
  if self.visited_nodes[b, action_idx] == 1.0:
      rewards[b] -= 5.0  # 재방문 시 강력한 패널티 부여
  ```
- 이 수정을 통해 워커는 `visited_nodes` (5번째 채널)을 보고 해당 노드를 피해야 한다는 것을 PPO 그라디언트를 통해 배우게 됩니다.

### 1.2 `src/envs/hrl_env.py`
**Goal:** 매니저가 새로운 서브 목표(Subgoal Zone)를 제시할 때마다 워커의 금지 구역(Visited Memory)을 백지화합니다.
- 사용자의 피드백 반영: "이미 방문한 노드를 영구히 막으면, 나중에 다른 목적지로 갈 때 지나가야 하는 길목이 막혀버립니다."
- `step_manager()` 함수 내의 워커 턴 시작 부분 수정:
  ```python
  self.env.visited_nodes[b].zero_()
  self.env.visited_nodes[b, int(self.env.curr_nodes[b].item())] = 1.0
  ```
- 임시로 추가했던 `visited_mask * 1e5` Logit 페널티를 완전히 롤백하여 워커가 자율적으로 결정하도록 복구합니다.

## 2. Training Strategy
- `scripts/train_rl.py --stage worker --episodes 10000` 재실행.
- 워커는 로컬한 단위에서만 학습하므로, Manager와 달리 10,000 에피소드만으로도 완벽하게 최적 경로(Loop-free)를 체득할 수 있습니다.
- 워커 학습이 끝난 후, 이미 확보된 최고의 매니저 가중치(`best_manager.pt`)와 결합하여 검증합니다.
