import torch
import sys
import os

sys.path.insert(0, '.')
from src.envs.worker_env import WorkerEnv
from src.envs.hrl_env import HRLEnv
from src.models.worker import Worker
from src.models.manager_unified import ManagerUnified

def get_env_and_manager():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    map_name = 'Anaheim'
    
    worker_env = WorkerEnv(
        f'data/{map_name}_node.tntp', f'data/{map_name}_net.tntp',
        zone_json=f'data/grid_{map_name}_node_to_zone.json',
        zone_graph_json=f'data/grid_{map_name}_zone_graph.json',
        masking_mode='soft_curr_next',
        disaster_prob=0.0,
        dynamic_disaster=False,
        device=device
    )
    
    worker = Worker(node_dim=5, hidden_dim=256, num_layers=2, dropout=0.0).to(device)
    hrl_env = HRLEnv(worker, worker_env)
    
    manager = ManagerUnified(
        zone_dim=6, target_dim=4, hidden_dim=128,
        num_gat_layers=2, gat_heads=4, num_transformer_layers=1, transformer_heads=4
    ).to(device)
    
    return hrl_env, manager, device

def test_manager_distance_heuristic(env_and_manager):
    hrl_env, manager, device = env_and_manager
    state = hrl_env.reset(batch_size=2)
    
    scene = hrl_env._get_scenario_dict()
    zf = scene['zone_features']
    zei = scene['zone_edge_index']
    tz = scene['target_zones']
    z_dist_matrix = scene['zone_dist_matrix']
    
    assert z_dist_matrix.shape == (hrl_env.env.k, hrl_env.env.k), "zone_dist_matrix shape is incorrect"
    assert z_dist_matrix.device.type == device.type, f"device mismatch: {z_dist_matrix.device} vs {device}"
    
    B = 2
    K = hrl_env.env.k
    flat_zf = zf.view(B * K, 6)
    ai = torch.arange(B, device=device).repeat_interleave(K)
    
    zone_emb = manager.encode_zones(flat_zf, zei, batch=ai)
    assert zone_emb.shape == (B * K, 128)
    
    # Mock parameters for get_zone_logits
    query = torch.randn(B, 128, device=device)
    selected_target_emb = torch.randn(B, 128, device=device)
    zone_adj_mask = torch.ones(B, K, device=device)
    zone_batch = torch.arange(B, device=device).repeat_interleave(K)
    
    selected_target_zone_idx = torch.tensor([0, K-1], device=device)
    mb_zone_dist_matrix = z_dist_matrix.unsqueeze(0).expand(B, -1, -1)
    
    logits, invalid = manager.get_zone_logits(
        query, selected_target_emb, zone_emb, zone_adj_mask.view(-1), zone_batch,
        selected_target_zone_idx, mb_zone_dist_matrix
    )
    
    assert logits.shape == (B, K)
    assert logits.shape == (B, K)
    assert invalid.shape == (B, K)
    print("Test passed: Distance heuristic tensor dimensional checks complete.")

if __name__ == '__main__':
    env_mgr = get_env_and_manager()
    test_manager_distance_heuristic(env_mgr)
