import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.manager_unified import ManagerUnified

def test_manager_unified_value_head():
    """TDD: ManagerUnified가 Value(Critic) 값을 올바른 차원[B]으로 반환하는지 검증"""
    B = 4
    K_zones = 10
    num_targets = 3
    hidden_dim = 128
    
    manager = ManagerUnified(zone_dim=6, target_dim=4, hidden_dim=hidden_dim, num_gat_layers=1, num_transformer_layers=1)
    
    # 더미 데이터 (배치 4, 구역 10, 타겟 3)
    zone_features = torch.randn(B * K_zones, 6)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    zone_batch = torch.arange(B).repeat_interleave(K_zones)
    
    target_features = torch.randn(B * num_targets, 4)
    target_zones = torch.randint(0, K_zones, (B * num_targets,))
    
    h_last = torch.randn(B, hidden_dim)
    elapsed = torch.rand(B, 1)
    rescued = torch.rand(B, 1)
    
    # 1. Zone Encoding
    zone_emb = manager.encode_zones(zone_features, edge_index, zone_batch)
    assert zone_emb.shape == (B * K_zones, hidden_dim), "Zone embedding shape mismatch"
    
    # 2. Context Generation
    query = manager.generate_context(h_last, elapsed, rescued)
    assert query.shape == (B, hidden_dim), "Query shape mismatch"
    
    # 3. Value Head (Critic)
    value = manager.get_value(query)
    assert value.shape == (B,), f"Value shape mismatch. Expected ({B},), got {value.shape}"
    
    # 4. Target & Zone Logits
    target_mask = torch.ones(B * num_targets)
    target_batch = torch.arange(B).repeat_interleave(num_targets)
    
    act_offsets = torch.arange(B).repeat_interleave(num_targets) * K_zones
    flat_tz_idx = act_offsets + target_zones
    
    t_emb = manager.get_target_embeddings(zone_emb, target_features, flat_tz_idx)
    t_logits, _, t_emb_dense = manager.get_target_logits(query, t_emb, target_mask, target_batch)
    
    assert t_logits.shape == (B, num_targets), "Target logits shape mismatch"
    
    selected_t_act = torch.zeros(B, dtype=torch.long)
    selected_t_emb = t_emb_dense[torch.arange(B), selected_t_act]
    
    zone_adj_mask = torch.ones(B * K_zones)
    z_logits, _ = manager.get_zone_logits(query, selected_t_emb, zone_emb, zone_adj_mask, zone_batch)
    
    assert z_logits.shape == (B, K_zones), "Zone logits shape mismatch"

if __name__ == '__main__':
    test_manager_unified_value_head()
    print("All tests passed.")
