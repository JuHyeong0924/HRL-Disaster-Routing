import torch
from src.envs.worker_env import WorkerEnv
from src.envs.hrl_env import HRLEnv
from src.envs.disaster_map import DisasterMap

def test_worker_revisit_penalty():
    """Worker가 이미 방문한 노드를 선택할 경우 -5.0 패널티를 받는지 검증합니다."""
    device = torch.device('cpu')
    env = WorkerEnv(
        node_file='data/Anaheim_node.tntp',
        net_file='data/Anaheim_net.tntp',
        zone_json='data/grid_Anaheim_node_to_zone.json',
        zone_graph_json='data/grid_Anaheim_zone_graph.json',
        device=device
    )
    
    env.reset(batch_size=1)
    
    # 1. 특정 노드를 방문 처리
    curr_node_idx = int(env.curr_nodes[0].item())
    adj_nodes = env._adj_list[curr_node_idx]
    
    if len(adj_nodes) == 0:
        print("Skipped: No adjacent nodes to test")
        return
        
    target_action = adj_nodes[0]
    
    # 해당 노드를 임의로 방문 처리
    env.visited_nodes[0, target_action] = 1.0
    
    # 2. 강제로 해당 노드로 스텝 진행
    action_tensor = torch.tensor([target_action], dtype=torch.long, device=device)
    _, rewards, _, _ = env.step_batch(action_tensor)
    
    # 3. 보상 검증 (-5.0 패널티가 적용되었는지 확인)
    assert rewards[0].item() <= -4.0, f"Expected strong negative penalty for revisit, but got {rewards[0].item()}"

def test_hrl_visited_reset_per_manager_turn():
    """Manager 턴이 바뀔 때 Worker의 visited_nodes가 초기화되는지 검증합니다."""
    device = torch.device('cpu')
    worker_env = WorkerEnv(
        node_file='data/Anaheim_node.tntp',
        net_file='data/Anaheim_net.tntp',
        zone_json='data/grid_Anaheim_node_to_zone.json',
        zone_graph_json='data/grid_Anaheim_zone_graph.json',
        device=device
    )
    from src.models.worker import Worker
    worker = Worker(node_dim=5, hidden_dim=64, num_layers=1, dropout=0.0).to(device)
    
    env = HRLEnv(worker=worker, worker_env=worker_env)
    obs = env.reset(batch_size=1, num_targets=2)
    
    # 1. 워커 환경에 임의의 쓰레기(방문 이력) 주입
    env.env.visited_nodes[0].fill_(1.0)
    
    # 2. 매니저 스텝 진행 (새로운 턴 시작)
    target_action = torch.tensor([0], dtype=torch.long, device=device)
    zone_action = torch.tensor([0], dtype=torch.long, device=device)
    env.step_manager(target_action, zone_action)
    
    # 3. 방문 이력 초기화 검증 (리셋 후, 이번 턴에서 Worker가 걸어간 스텝 수 + 출발 위치만큼 1.0이어야 함)
    visited_sum = env.env.visited_nodes[0].sum().item()
    expected_sum = float(env.env.steps_count[0].item() + 1)
    
    assert visited_sum <= expected_sum, f"visited_nodes was not properly reset. Sum is {visited_sum}, expected <= {expected_sum}"
    
    curr_idx = int(env.env.curr_nodes[0].item())
    assert env.env.visited_nodes[0, curr_idx].item() == 1.0, "Current node must be marked as visited."

if __name__ == '__main__':
    test_worker_revisit_penalty()
    test_hrl_visited_reset_per_manager_turn()
    print("✅ All TDD tests passed successfully!")
