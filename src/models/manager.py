import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GraphNorm
from torch_geometric.utils import to_dense_batch

class Manager(nn.Module):
    """
    Zone-Aware Target & Route Selector.
    
    GATv2로 Zone Graph의 로컬 토폴로지를 인코딩하고,
    Transformer Self-Attention으로 전역 컨텍스트를 추가한 뒤,
    Dual Head 디코더로 Target과 Zone을 단계적으로 선택합니다.
    """
    def __init__(self, zone_dim=7, target_dim=6, hidden_dim=256,
                 num_gat_layers=3, gat_heads=4, num_transformer_layers=3,
                 transformer_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Zone Encoder (GATv2)
        self.zone_proj = nn.Linear(zone_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(num_gat_layers):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, concat=False, dropout=dropout)
            )
            self.gat_norms.append(GraphNorm(hidden_dim))
            
        # 2. Transformer Global Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=transformer_heads, 
            dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers,
            enable_nested_tensor=False
        )
        
        # 3. Target Fusion MLP
        self.target_fusion = nn.Sequential(
            nn.Linear(hidden_dim + target_dim, hidden_dim),  # 256+6=262
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 4. Context Generator
        # [Phase 2B] h_last 제거 (항상 0 → 노이즈)
        # 입력: elapsed_time(1) + num_rescued(1) + num_feasible(1) + avg_urgency(1) = 4
        self.context_proj = nn.Linear(4, hidden_dim)
        
        # 5. Zone Score Network (Query + ZoneEmb + Dist)
        self.zone_score_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 6. Critic Head (Value Estimator)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode_zones(self, zone_features, zone_edge_index, batch=None):
        """
        zone_features: [total_K, 6]
        zone_edge_index: [2, E]
        batch: [total_K] (PyG batch indicator)
        Returns: [total_K, 128]
        """
        x = self.zone_proj(zone_features)
        
        for conv, norm in zip(self.gat_layers, self.gat_norms):
            x_res = x
            x = conv(x, zone_edge_index)
            if batch is not None:
                x = norm(x, batch)
            else:
                x = norm(x)
            x = torch.relu(x)
            x = x + x_res
            
        if batch is not None:
            x_dense, mask = to_dense_batch(x, batch)  # [B, max_K, 128], [B, max_K]
        else:
            x_dense = x.unsqueeze(0)  # [1, K, 128]
            mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
            
        # Transformer는 padding mask를 ~mask 로 받음 (True가 무시할 곳)
        x_global = self.transformer(x_dense, src_key_padding_mask=~mask)
        
        x_out = x_global[mask]  # [total_K, 128]
        return x_out
        
    def get_target_embeddings(self, zone_embeddings, target_features, target_zone_idx):
        """
        zone_embeddings: [total_K, 128]
        target_features: [total_N, 6]  # [Phase 2A] 4→6
        target_zone_idx: [total_N] (flattened index matching zone_embeddings)
        Returns: [total_N, 128]
        """
        tz_emb = zone_embeddings[target_zone_idx]
        concat = torch.cat([tz_emb, target_features], dim=-1)  # [total_N, 128+6=134]
        return self.target_fusion(concat)
        
    def generate_context(self, elapsed_time, num_rescued, num_feasible, avg_urgency):
        """
        [Phase 2B] h_last 제거, 의미 있는 스칼라 피처 4개로 대체.
        elapsed_time: [B, 1]   — 경과 시간 (정규화)
        num_rescued: [B, 1]    — 구출 수 (정규화)
        num_feasible: [B, 1]   — 도달 가능 타겟 비율
        avg_urgency: [B, 1]    — 평균 긴급도
        Returns: query [B, 128]
        """
        ctx_cat = torch.cat([elapsed_time, num_rescued, num_feasible, avg_urgency], dim=-1)  # [B, 4]
        return self.context_proj(ctx_cat)
        
    def get_target_logits(self, query, target_embeddings, target_mask, target_batch):
        """
        query: [B, 128]
        target_embeddings: [total_N, 128]
        target_mask: [total_N] (1 for valid, 0 for invalid/rescued/timeout)
        target_batch: [total_N]
        
        Returns: [B, max_N] logits, and the invalid mask [B, max_N]
        """
        t_emb_dense, mask_dense = to_dense_batch(target_embeddings, target_batch) # [B, max_N, 128]
        t_mask_valid, _ = to_dense_batch(target_mask, target_batch) # [B, max_N]
        
        scores = torch.bmm(query.unsqueeze(1), t_emb_dense.transpose(1, 2)).squeeze(1) # [B, max_N]
        scores = scores / (self.hidden_dim ** 0.5)
        
        invalid = (~mask_dense) | (t_mask_valid == 0)
        scores.masked_fill_(invalid, float('-inf'))
        return scores, invalid, t_emb_dense
        
    def get_zone_logits(self, query, selected_target_emb, zone_embeddings, zone_adj_mask, zone_batch, selected_target_zone_idx, zone_dist_matrix):
        """
        query: [B, 128]
        selected_target_emb: [B, 128]
        zone_embeddings: [total_K, 128]
        zone_adj_mask: [total_K] (1 for valid adjacent zones, 0 for invalid)
        zone_batch: [total_K]
        selected_target_zone_idx: [B]
        zone_dist_matrix: [B, K, K]
        
        Returns: [B, max_K] logits, and the invalid mask [B, max_K]
        """
        z_emb_dense, mask_dense = to_dense_batch(zone_embeddings, zone_batch) # [B, max_K, 128]
        z_mask_valid, _ = to_dense_batch(zone_adj_mask, zone_batch) # [B, max_K]
        
        B, max_K, _ = z_emb_dense.shape
        
        # 1. Expand query & selected_target_emb to match max_K
        query_expanded = query.unsqueeze(1).expand(B, max_K, -1) # [B, max_K, 128]
        
        # 2. Extract distance from each Zone to the Selected Target Zone
        # dist_matrix: [B, K, K]. We want [B, K] where it's the distance to selected_target_zone_idx.
        b_idx = torch.arange(B, device=query.device)
        target_dists = zone_dist_matrix[b_idx, :, selected_target_zone_idx].unsqueeze(-1) # [B, K, 1]
        
        # Pad target_dists if max_K > K (although K is usually constant per batch)
        if max_K > target_dists.size(1):
            pad = torch.zeros(B, max_K - target_dists.size(1), 1, device=query.device)
            target_dists = torch.cat([target_dists, pad], dim=1)
            
        # 3. Concatenate and score
        combined = torch.cat([query_expanded, z_emb_dense, target_dists], dim=-1) # [B, max_K, 128 + 128 + 1]
        scores = self.zone_score_net(combined).squeeze(-1) # [B, max_K]
        
        invalid = (~mask_dense) | (z_mask_valid == 0)
        scores.masked_fill_(invalid, float('-inf'))
        return scores, invalid

    def get_value(self, query):
        """
        query: [B, 128] generated from generate_context
        Returns: [B] state value
        """
        return self.value_head(query).squeeze(-1)
