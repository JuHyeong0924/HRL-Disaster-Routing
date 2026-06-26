"""
Phase 2 Manager 정확도 개선 테스트: Target Features, Manager 모델 구조, 보상 재설계

검증 항목:
1. Target Features [B, N, 6] 차원 및 채널 의미 검증
2. Manager 모델 forward pass (target_dim=6, context 4-dim) 텐서 흐름
3. generate_context 시그니처 변경 (h_last 제거)
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.manager import Manager


class TestManagerModel:
    """Phase 2B: Manager 모델 구조 변경 검증."""
    
    def test_manager_init_target_dim_6(self):
        """Manager 기본 생성 시 target_dim=6인지 확인."""
        m = Manager()
        # target_fusion 첫 레이어 입력: hidden_dim(256) + target_dim(6) = 262
        first_layer = m.target_fusion[0]
        assert first_layer.in_features == 256 + 6, \
            f"target_fusion input dim이 262여야 합니다: {first_layer.in_features}"
    
    def test_context_proj_dim_4(self):
        """context_proj 입력이 4차원인지 확인 (h_last 제거)."""
        m = Manager(hidden_dim=128)
        assert m.context_proj.in_features == 4, \
            f"context_proj input dim이 4여야 합니다: {m.context_proj.in_features}"
        assert m.context_proj.out_features == 128
    
    def test_generate_context_no_h_last(self):
        """generate_context에서 h_last 없이 4개 스칼라로 query 생성."""
        m = Manager(hidden_dim=128)
        B = 4
        elapsed = torch.rand(B, 1)
        rescued = torch.rand(B, 1)
        num_feasible = torch.rand(B, 1)
        avg_urgency = torch.rand(B, 1)
        
        query = m.generate_context(elapsed, rescued, num_feasible, avg_urgency)
        assert query.shape == (B, 128), f"query shape이 [B, 128]이어야 합니다: {query.shape}"
    
    def test_target_embeddings_dim_6(self):
        """target_features [total_N, 6]으로 get_target_embeddings 통과."""
        m = Manager(hidden_dim=128)
        total_K = 10
        total_N = 5
        
        zone_emb = torch.rand(total_K, 128)
        target_features = torch.rand(total_N, 6)  # 6-dim
        target_zone_idx = torch.randint(0, total_K, (total_N,))
        
        t_emb = m.get_target_embeddings(zone_emb, target_features, target_zone_idx)
        assert t_emb.shape == (total_N, 128), f"target emb shape: {t_emb.shape}"
    
    def test_full_forward_pass(self):
        """Manager 전체 forward pass (zone encoding → target emb → context → logits)."""
        m = Manager(hidden_dim=128, num_gat_layers=2, num_transformer_layers=2)
        m.eval()
        
        B, K, N = 2, 8, 5
        
        # Zone features + edges
        zone_features = torch.rand(B * K, 6)
        zone_batch = torch.arange(B).repeat_interleave(K)
        # Simple ring graph per batch
        edges = []
        for b in range(B):
            offset = b * K
            for i in range(K):
                edges.append([offset + i, offset + (i + 1) % K])
                edges.append([offset + (i + 1) % K, offset + i])
        zone_edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Target features
        target_features = torch.rand(B * N, 6)
        target_zone_idx = torch.randint(0, K, (B * N,))
        # Adjust to batch offsets
        target_batch_offsets = torch.arange(B).repeat_interleave(N) * K
        target_zone_idx_flat = target_batch_offsets + target_zone_idx
        target_mask = torch.ones(B * N, dtype=torch.long)
        target_batch = torch.arange(B).repeat_interleave(N)
        
        # Context
        elapsed = torch.rand(B, 1)
        rescued = torch.rand(B, 1)
        num_feasible = torch.rand(B, 1)
        avg_urgency = torch.rand(B, 1)
        
        with torch.no_grad():
            zone_emb = m.encode_zones(zone_features, zone_edge_index, batch=zone_batch)
            assert zone_emb.shape == (B * K, 128)
            
            t_emb = m.get_target_embeddings(zone_emb, target_features, target_zone_idx_flat)
            assert t_emb.shape == (B * N, 128)
            
            query = m.generate_context(elapsed, rescued, num_feasible, avg_urgency)
            assert query.shape == (B, 128)
            
            t_logits, _, t_emb_dense = m.get_target_logits(query, t_emb, target_mask, target_batch)
            assert t_logits.shape[0] == B
            
            value = m.get_value(query)
            assert value.shape == (B,)
