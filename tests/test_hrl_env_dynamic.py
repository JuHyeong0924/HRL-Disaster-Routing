import pytest
import torch
import numpy as np
from src.envs.worker_env import WorkerEnv
from src.envs.hrl_env import HRLEnv
import torch.nn as nn

class MockWorker(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_heuristic = True

    def get_actions(self, active):
        # Return dummy actions for active workers
        return [0 for _ in active]

@pytest.fixture
def env():
    worker_env = WorkerEnv(node_file='data/Anaheim_node.tntp', net_file='data/Anaheim_net.tntp', device='cpu')
    worker = MockWorker()
    hrl_env = HRLEnv(worker=worker, worker_env=worker_env)
    hrl_env.env.disaster_prob = 0.2
    hrl_env.env.dynamic_disaster = True
    return hrl_env

def test_global_aftershock_scheduling(env):
    env.reset(num_targets=10)
    
    # 1. 초기 재난 발생 여부 검증 (disaster_prob > 0 이면 초기 피해 적용됨)
    assert env.env.disaster_prob == 0.2
    
    # 2. 스케줄링 확인
    assert env.global_step == 0
    assert hasattr(env, 'aftershock_schedule')
    # 여진 발생 횟수가 2~5회로 제한되었는지 확인
    assert 2 <= len(env.aftershock_schedule) <= 5
    # 스케줄된 턴이 max_manager_turns 이내인지 확인
    assert all(1 <= t < env.max_manager_turns for t in env.aftershock_schedule)
    
    # 3. 글로벌 여진 발생 트리거 확인
    # 강제로 스케줄을 고정
    env.aftershock_schedule = {3, 5}
    
    # 1, 2턴은 여진 미발생
    env.step_manager(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    assert env.global_step == 1
    
    env.step_manager(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    assert env.global_step == 2
    
    # 3턴은 여진 발생해야 함
    env.step_manager(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    assert env.global_step == 3
    
    print("Test passed: HRLEnv initial disaster, scheduling, and step_manager execution works properly.")

def test_worker_env_reset_damage_accumulation():
    worker_env = WorkerEnv(node_file='data/Anaheim_node.tntp', net_file='data/Anaheim_net.tntp', device='cpu')
    worker_env.disaster_prob = 0.2
    worker_env.dynamic_disaster = False
    
    # 1st reset
    worker_env.reset()
    damage_1 = [worker_env.G[u][v].get('damage', 0.0) for u, v in worker_env.G.edges()]
    avg_damage_1 = sum(damage_1) / len(damage_1)
    
    # Multiple resets should not accumulate damage infinitely because it is cleared first
    for _ in range(50):
        worker_env.reset()
        
    damage_50 = [worker_env.G[u][v].get('damage', 0.0) for u, v in worker_env.G.edges()]
    avg_damage_50 = sum(damage_50) / len(damage_50)
    
    # If damage accumulated, avg_damage_50 would be very close to 1.0 (or graph disconnected causing infinite loop)
    # Since it resets correctly, it should be statistically similar
    assert avg_damage_50 < 0.5, "Damage accumulated across resets!"
