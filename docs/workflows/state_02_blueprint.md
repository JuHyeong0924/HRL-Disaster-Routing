# State 02: Blueprint (Manager Stagnation Hotfix)

## 1. Architectural Strategy
이 청사진은 매니저(Manager)가 동일한 구역(Zone)을 무의미하게 연속해서 지시하여 발생하는 '라우팅 붕괴(Routing Collapse)' 현상(전체의 약 20%)을 해결하기 위한 2가지 조화로운 강화학습 튜닝 기법을 정의합니다. 

본 설계는 코어 아키텍처(`hrl_env.py` 등)의 수정을 피하고, 순수하게 훈련 루프(`train_manager.py`) 단에서 보상과 하이퍼파라미터만 주입하여 시스템 안정성을 100% 보장합니다.

## 2. Dynamic PPO Entropy Decay (탐험성 보장)
- **개념**: 매니저가 잘못된 판단에 대해 100%의 확률적 확신(Over-confidence)을 가지는 현상을 방지합니다.
- **수정 대상**: `scripts/train_manager.py`의 `ManagerPPOTrainer` 클래스.
- **구현 방식**:
  - 기존: `loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy` (상수 0.01)
  - 변경: `ent_coef` 파라미터를 `train_step()` 함수의 인자로 추가.
  - 감가(Decay) 스케줄링: 학습 초기에는 `0.05`에서 시작하여 10,000 에피소드에 걸쳐 서서히 `0.01`로 떨어지는 선형 감소 함수 적용.
    ```python
    current_ent_coef = max(0.01, 0.05 - 0.04 * (ep / 10000.0))
    ```

## 3. Manager Stagnation Penalty (정체 감점 조형)
- **개념**: 직전 턴에 지시했던 구역과 완벽히 동일한 구역을 지시했는데, 그 턴 동안 워커가 아무런 타겟도 구조하지 못했다면(무의미한 턴 낭비), 매니저에게 **-2.0점의 즉각적인 패널티**를 부여합니다.
- **수정 대상**: `scripts/train_manager.py` 메인 시뮬레이션 루프
- **구현 방식**:
  - `prev_manager_zones = torch.full((batch_size,), -1, dtype=torch.long, device=device)` 텐서를 선언하여 각 턴마다 매니저가 선택한 `z_act`를 기억합니다.
  - 한 턴이 끝난 후 보상을 계산할 때 다음 논리회로를 추가합니다:
    ```python
    stagnation_mask = (z_act == prev_manager_zones) & (~rescued_this_turn)
    stagnation_penalty = stagnation_mask.float() * -2.0
    step_reward = step_reward + stagnation_penalty
    prev_manager_zones = z_act.clone()
    ```

## 4. Expected Outcomes
1. **무한 루프 원천 차단**: `-2.0`점의 지속적인 누적 감점을 피하기 위해 매니저는 PPO 정책 업데이트 과정에서 자연스럽게 "같은 구역을 다시 고르는 행동"의 가치를 깎아내리게 됩니다.
2. **Entropy 시너지**: Entropy 계수가 높아진 상태이므로, 패널티를 받은 매니저는 손쉽게 플랜 B(두 번째로 높은 로짓을 가진 구역)로 우회하는 법을 체득하게 됩니다.
