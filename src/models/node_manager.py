import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MANAGER_DECODE_BIAS_CFG = {
    'corridor_bonus': 1.75,
    'progress_bonus_scale': 1.75,
    'detour_penalty_scale': 0.45,
    'nonprogress_penalty': 2.00,
    'eos_near_goal_bonus': 0.75,
    'eos_far_penalty': 1.25,
    'eos_far_penalty_scale': 0.35,
    'eos_far_penalty_cap': 8.0,
    'target_segment_hops': 4.5,
    'plan_count_band': 1,
    'hard_eos_min_hops': 4.0,
    'corridor_slack': 2.0,
    'eos_near_goal_hops': 1.0,
    'radius_low_penalty_scale': 0.75,
    'directional_penalty_scale': 1.25,
    'directional_cos_margin': -0.25,
}


def compute_manager_decode_bias(
    apsp_matrix,
    start_idx,
    current_idx,
    goal_idx,
    eos_index,
    cfg=None,
    generated_tokens_so_far=0,
):
    """
    Soft goal-aware decode bias shared by RL training and rollout visualization.

    Returns:
        dict with [N+1] tensors for total bias and each component, plus scalar hop stats.
    """
    cfg = DEFAULT_MANAGER_DECODE_BIAS_CFG if cfg is None else cfg
    device = apsp_matrix.device
    num_nodes = int(apsp_matrix.size(0))

    shortest_hops = float(apsp_matrix[start_idx, goal_idx].item())
    if not torch.isfinite(torch.tensor(shortest_hops)):
        shortest_hops = 1.0
    shortest_hops = max(shortest_hops, 1.0)

    current_goal_hops = float(apsp_matrix[current_idx, goal_idx].item())
    if not torch.isfinite(torch.tensor(current_goal_hops)):
        current_goal_hops = shortest_hops

    hops_from_start = apsp_matrix[start_idx, :num_nodes].float()
    hops_to_goal = apsp_matrix[:num_nodes, goal_idx].float()
    total_via_hops = hops_from_start + hops_to_goal

    corridor_limit = shortest_hops + float(cfg['corridor_slack'])
    corridor_ok = torch.isfinite(total_via_hops) & (total_via_hops <= corridor_limit)
    corridor_bonus = corridor_ok.float() * float(cfg['corridor_bonus'])

    progress_bonus = (
        (current_goal_hops - hops_to_goal) / max(current_goal_hops, 1.0)
    ) * float(cfg['progress_bonus_scale'])

    detour_excess = torch.clamp(total_via_hops - corridor_limit, min=0.0)
    detour_penalty = -float(cfg['detour_penalty_scale']) * detour_excess

    nonprogress_mask = hops_to_goal >= (current_goal_hops - 1e-6)
    nonprogress_penalty = -float(cfg['nonprogress_penalty']) * nonprogress_mask.float()

    full_shape = (num_nodes + 1,)
    full_corridor_bonus = torch.zeros(full_shape, device=device)
    full_progress_bonus = torch.zeros(full_shape, device=device)
    full_detour_penalty = torch.zeros(full_shape, device=device)
    full_nonprogress_penalty = torch.zeros(full_shape, device=device)
    full_eos_bias = torch.zeros(full_shape, device=device)

    full_corridor_bonus[:-1] = corridor_bonus
    full_progress_bonus[:-1] = progress_bonus
    full_detour_penalty[:-1] = detour_penalty
    full_nonprogress_penalty[:-1] = nonprogress_penalty
    target_segment_hops = max(float(cfg.get('target_segment_hops', 4.5)), 1.0)
    k_ref = max(1.0, float(math.ceil(shortest_hops / target_segment_hops)))
    k_min = 1.0 if shortest_hops <= float(cfg.get('hard_eos_min_hops', 4.0)) else k_ref
    eos_hard_masked = False
    if current_goal_hops <= float(cfg['eos_near_goal_hops']):
        full_eos_bias[eos_index] = float(cfg['eos_near_goal_bonus'])
    else:
        extra_far_hops = max(current_goal_hops - float(cfg['eos_near_goal_hops']), 0.0)
        extra_far_hops = min(extra_far_hops, float(cfg.get('eos_far_penalty_cap', 8.0)))
        if float(generated_tokens_so_far) < k_min:
            full_eos_bias[eos_index] = float('-inf')
            eos_hard_masked = True
        else:
            full_eos_bias[eos_index] = -(
                float(cfg['eos_far_penalty'])
                + float(cfg.get('eos_far_penalty_scale', 0.0)) * extra_far_hops
            )

    total_bias = (
        full_corridor_bonus
        + full_progress_bonus
        + full_detour_penalty
        + full_nonprogress_penalty
        + full_eos_bias
    )
    return {
        'total_bias': total_bias,
        'corridor_bonus': full_corridor_bonus,
        'progress_bonus': full_progress_bonus,
        'detour_penalty': full_detour_penalty,
        'nonprogress_penalty': full_nonprogress_penalty,
        'eos_bonus_or_penalty': full_eos_bias,
        'shortest_hops': shortest_hops,
        'current_goal_hops': current_goal_hops,
        'plan_len_ref': k_ref,
        'plan_len_min': k_min,
        'eos_hard_masked': eos_hard_masked,
    }


class GraphTransformerManager(nn.Module):
    """
    Manager: Graph Transformer Encoder + Transformer Decoder.
    
    아키텍처 변경 (v2):
    - [기존] GNN(GATConv) Encoder → global_mean_pool → 단일 Context Vector
    - [개선] Transformer Encoder → 노드 수준 임베딩 전체를 Decoder Memory로 전달
    
    핵심 개선점:
    1. GNN 제거: GATConv의 로컬 수용장(3-hop) 한계 극복
    2. 전역 Self-Attention: 모든 노드 간 관계를 한 번에 계산
    3. 정보 병목 해소: Decoder가 개별 노드에 Cross-Attention 가능
       (기존: 전체 그래프를 단일 벡터로 압축 → 시퀀스 생성 성능 저하)
    4. 학습 가능한 노드 위치 임베딩: 고정 그래프의 위상 구조를 암묵적으로 학습
    """

    def __init__(self, node_dim, hidden_dim, num_layers=3, heads=4, dropout=0.2, edge_dim: int = 5):
        """
        Args:
            node_dim: 노드 피처 차원 (4: [x, y, is_start, is_goal])
            hidden_dim: 은닉층 차원
            num_layers: Transformer 레이어 수
            heads: Multi-Head Attention 수
            dropout: Dropout 비율
            edge_dim: 엣지 피처 차원 (Phase 2 대비)
        """
        super(GraphTransformerManager, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        # [Constants]
        self.PAD_TOKEN = -100
        self.EOS_TOKEN = -2 # Placeholder, actual EOS is dynamic (N)
        
        # [Topology-Aware Refactoring]
        # 1. GATv2Conv for Initial Node Embedding (Captures Local Topology)
        from torch_geometric.nn import GATv2Conv
        self.topology_enc = GATv2Conv(node_dim, hidden_dim, heads=4, concat=False, dropout=dropout, edge_dim=edge_dim)
        
        # 2. Laplacian PE Configurations (Additive)
        self.k_eig = 8 
        self.lpe_dim = hidden_dim # Project to hidden_dim
        self.lpe_lin = nn.Linear(self.k_eig, self.lpe_dim)
        
        # [Safety] Initialize LPE Cache
        self._cached_pe = None
        self._cached_lpe_N = None
        
        # [Stabilization] LayerNorm for Input Embeddings (GAT + LPE)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        
        # 3. Transformer Encoder (Global View)
        # No Masking -> Full Self-Attention (Global Reasoning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        
        # === Decoder (Pointer Network) ===
        self.sos_emb = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.eos_token_emb = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Pointer Attention
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_v = nn.Parameter(torch.randn(hidden_dim, 1))
        self.decode_bias_cfg = dict(DEFAULT_MANAGER_DECODE_BIAS_CFG)
        
        # [Initialization] Stabilize Pointer Network Training
        nn.init.xavier_uniform_(self.pointer_v)
        nn.init.xavier_uniform_(self.pointer_query.weight)
        nn.init.xavier_uniform_(self.pointer_key.weight)


    def encode_graph(self, x, edge_index, batch, edge_attr=None):
        # 1. GATv2 Embedding (Local Topology + Edge Features)
        h_base = self.topology_enc(x, edge_index, edge_attr=edge_attr) # [B*N, H]
        
        # 2. Compute/Get LPE
        from torch_geometric.utils import to_dense_batch
        h_dense, mask = to_dense_batch(h_base, batch) # [B, N_max, H]
        B, N_max, H = h_dense.size()

        # --- LPE LOGIC FIX (User Requested) ---
        # Instead of complex dynamic calculation, let's assume strict consistency.
        # If the map is fixed, we calculate LPE *once* based on the edge_index of the *first graph*
        # and then repeat it.
        
        # Check if we need to re-compute (Only once per training session usually)
        # We use N_real to detect major changes or uninitialized state.
        
        # Extract meaningful node count of the first graph
        real_mask_0 = mask[0]
        N_real = real_mask_0.sum().long().item()
        
        if not hasattr(self, '_cached_pe') or self._cached_lpe_N != N_real:
             # Create dense adj for the first graph only
             from torch_geometric.utils import to_dense_adj
             # We can slice the edge_index to get only the first graph's edges
             # But simply using to_dense_adj on batch and taking [0] is safer/easier
             adj_full = to_dense_adj(edge_index, batch) 
             adj_0 = adj_full[0, :N_real, :N_real]
             
             # Laplacian Logic
             deg = adj_0.sum(dim=1)
             deg_inv_sqrt = deg.pow(-0.5)
             deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
             
             # L = I - D^-0.5 * A * D^-0.5
             L = torch.eye(N_real, device=x.device) - deg_inv_sqrt.unsqueeze(1) * adj_0 * deg_inv_sqrt.unsqueeze(0)
             
             # Eigendecomposition
             # Use eigh (Hermitian) for stability
             vals, vecs = torch.linalg.eigh(L)
             
             # Keep top-k (skip first trivial)
             # [N_real, k_eig]
             max_k = min(self.k_eig, N_real)
             if N_real < self.k_eig:
                 padding = torch.zeros(N_real, self.k_eig - N_real, device=x.device)
                 pe = torch.cat([vecs[:, :N_real], padding], dim=1)
             else:
                 pe = vecs[:, 1:self.k_eig+1] 
             
             # Cache Eigenvectors (Not projected embeddings)
             # This allows lpe_lin to be trained!
             self._cached_pe = pe 
             self._cached_lpe_N = N_real

        # Project and Expand
        # [N_real, k] -> [N_real, H]
        pe_emb = self.lpe_lin(self._cached_pe)
        
        # Expand cached PE to batch size
        # We have pe_emb: [N_real, H]
        # We need [B, N_max, H]
        
        # Create a zero tensor for the full dense shape [B, N_max, H]
        # This handles the padding automatically (zeros for padded nodes)
        pe_batch = torch.zeros((B, N_max, H), device=x.device)
        
        # Fill the valid region for ALL batches
        # We assume every graph in the batch is identical to the first one.
        # So we broadcast [N_real, H] -> [B, N_real, H]
        pe_batch[:, :N_real, :] = pe_emb.unsqueeze(0).expand(B, -1, -1)
        
        # Add to node features
        h_with_pe = h_dense + pe_batch
        
        # [Stabilization] LayerNorm before Transformer
        # GAT output and LPE projection might have different scales.
        h_with_pe = self.embedding_norm(h_with_pe)
        
        # 3. Mask for Padding (Float)
        # [Note] Use Float Mask (-inf) to explicitly avoid PyTorch NestedTensor warning.
        # Newer PyTorch versions may prefer BoolTensor, but float is safer for now.
        padding_mask = torch.zeros(mask.size(), device=mask.device, dtype=torch.float)
        padding_mask.masked_fill_(~mask, float('-inf'))
        
        # 4. Encoder Forward (Full Transformer)
        h_encoded = self.encoder(h_with_pe, src_key_padding_mask=padding_mask)
        h_encoded = self.encoder_norm(h_encoded)
        
        return h_encoded, mask

    def forward(self, x, edge_index, batch, target_seq_emb=None, edge_attr=None):
        memory, memory_mask = self.encode_graph(x, edge_index, batch, edge_attr=edge_attr) 
        batch_size = memory.size(0)
        
        # [Inductive] Append EOS Token to Memory (Virtual Node)
        # memory: [B, N, H] -> [B, N+1, H]
        eos_node = self.eos_token_emb.expand(batch_size, -1, -1) # [B, 1, H]
        memory_extended = torch.cat([memory, eos_node], dim=1) # [B, N+1, H]
        
        # Update Mask: [B, N] -> [B, N+1] (EOS is always valid)
        eos_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        memory_mask_extended = torch.cat([memory_mask, eos_mask], dim=1)
        
        # Decoder Input
        sos = self.sos_emb.expand(batch_size, -1, -1)
        if target_seq_emb is not None:
            dec_input = torch.cat([sos, target_seq_emb], dim=1)
        else:
            dec_input = sos
            
        seq_len = dec_input.size(1)
        # [Fix] Positional Encoding 제거: VRP의 Permutation Invariant 특성 보존
        # 절대 위치 인코딩은 A* 출력 순서에 과적합(Order Bias)을 유발함
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        
        # [Fix Warning] Use Float Mask for Memory Key Padding Mask too
        padding_mask_float = torch.zeros(memory_mask_extended.size(), device=x.device, dtype=torch.float)
        padding_mask_float.masked_fill_(~memory_mask_extended, float('-inf'))
        
        dec_output = self.decoder(tgt=dec_input, memory=memory_extended, tgt_mask=tgt_mask, memory_key_padding_mask=padding_mask_float)
        
        # Pointer Attention
        # [AMP NaN Fix] Pointer Network의 tanh + matmul 연산에서 FP16 overflow가 발생하므로
        # 이 부분만 FP32로 격리하여 VRAM 절약과 수치 안정성을 동시에 확보
        # Q: dec_output [B, L, H]
        # K: memory_extended [B, N+1, H]
        with torch.amp.autocast('cuda', enabled=False):
            dec_f32 = dec_output.float()
            mem_f32 = memory_extended.float()
            Q = self.pointer_query(dec_f32).unsqueeze(2) # [B, L, 1, H]
            K = self.pointer_key(mem_f32).unsqueeze(1) # [B, 1, N+1, H]
            
            attn_logits = torch.tanh(Q + K)
            attn_logits = torch.matmul(attn_logits, self.pointer_v.float()).squeeze(-1) # [B, L, N+1]
            
            # Masking (Use Bool mask here for simple masking)
            # memory_mask_extended is Bool, True where Valid.
            # So ~memory_mask_extended is True where Invalid.
            expanded_mask = (~memory_mask_extended).unsqueeze(1).expand_as(attn_logits)
            attn_logits.masked_fill_(expanded_mask, float('-inf'))
        
        return attn_logits # [B, L, N+1] (Last index is EOS)

    @torch.no_grad()
    def generate(self, x, edge_index, batch, valid_tokens=None, max_len=20, temperature=1.0, apsp_matrix=None, node_positions=None, edge_attr=None):
        memory, memory_mask = self.encode_graph(x, edge_index, batch, edge_attr=edge_attr)
        batch_size = memory.size(0)
        
        # Append EOS
        eos_node = self.eos_token_emb.expand(batch_size, -1, -1)
        memory_extended = torch.cat([memory, eos_node], dim=1)
        eos_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        memory_mask_extended = torch.cat([memory_mask, eos_mask], dim=1)
        
        curr_emb = self.sos_emb.expand(batch_size, -1, -1)
        full_seqs = []
        visited_mask = torch.zeros_like(memory_mask_extended, dtype=torch.bool) # [B, N+1]
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        
        # N is the index of EOS
        EOS_INDEX = memory.size(1) 
        
        for k in range(max_len):
            # [Fix] Positional Encoding 제거: Permutation Invariant 특성 보존
            curr_input = curr_emb
            
            # [Fix Warning] Float Mask
            padding_mask_float = torch.zeros(memory_mask_extended.size(), device=x.device, dtype=torch.float)
            padding_mask_float.masked_fill_(~memory_mask_extended, float('-inf'))
            
            # Decode using Extended Memory
            out = self.decoder(tgt=curr_input, memory=memory_extended, memory_key_padding_mask=padding_mask_float)
            
            # [Fix] Explicit Broadcasting (Same as Forward)
            Q = self.pointer_query(out).unsqueeze(2)        # [B, 1, 1, H]
            K = self.pointer_key(memory_extended).unsqueeze(1) # [B, 1, N+1, H]
            
            scores = torch.tanh(Q + K) # [B, 1, N+1, H]
            scores = torch.matmul(scores, self.pointer_v).squeeze(-1).squeeze(1) # [B, 1, N+1, 1] -> [B, N+1]
            if scores.dim() == 3: scores = scores.squeeze(1)
            
            # Masking
            scores.masked_fill_(~memory_mask_extended, float('-inf'))
            
            # --- 1. 과거 궤적 (Visited) 마스킹 ---
            scores.masked_fill_(visited_mask, float('-inf'))
            
            # 아래 마스킹 로직들은 노드 클래스(0 ~ N-1)에만 적용되며, EOS 인덱스(N)에는 영향이 없어야 합니다.
            # scores shape: [B, N+1]
            
            if (apsp_matrix is not None) and (node_positions is not None):
                # 타겟 노드(목적지) 찾기: x[..., 3] == 1.0
                # 이 배치의 현재 상태(직전 예측 노드)
                curr_nodes = next_node_indices if k > 0 else None
                
                for b_i in range(batch_size):
                    # 1. 목적지 노드 탐색
                    # batch == b_i 인 노드들 중 x[i, 3] == 1 인 노드
                    b_mask = (batch == b_i)
                    b_x = x[b_mask]
                    start_idx = torch.argmax(b_x[:, 2]).item()
                    goal_idx = torch.argmax(b_x[:, 3]).item() # local idx
                    
                    # 2. 현재 위치 노드 탐색
                    if curr_nodes is not None:
                        c_idx = curr_nodes[b_i].item()
                    else:
                        c_idx = torch.argmax(b_x[:, 2]).item()
                        
                    # 만약 현재 위치가 EOS라면 건너뜀
                    if c_idx == EOS_INDEX:
                        continue
                        
                    # --- 2. 동적 segment budget 반경 제어 ---
                    hops_from_curr = apsp_matrix[c_idx].float()
                    shortest_hops = float(apsp_matrix[start_idx, goal_idx].item())
                    current_goal_hops = float(apsp_matrix[c_idx, goal_idx].item())
                    # inf 방어: 연결 불가능 노드 쌍인 경우 안전한 기본값 사용
                    if not math.isfinite(shortest_hops):
                        shortest_hops = 1.0
                    if not math.isfinite(current_goal_hops):
                        current_goal_hops = shortest_hops
                    target_segment_hops = max(float(self.decode_bias_cfg.get('target_segment_hops', 4.5)), 1.0)
                    k_ref = max(1.0, float(math.ceil(max(shortest_hops, 1.0) / target_segment_hops)))
                    remaining_slots = max(1.0, k_ref - float(k))
                    seg_ref = max(current_goal_hops / remaining_slots, 1.0)
                    r_min = max(2.0, float(math.floor(0.5 * seg_ref)))
                    r_max = float(math.ceil(1.5 * seg_ref) + 1.0)

                    radius_mask_high = hops_from_curr > r_max
                    radius_mask_high[goal_idx] = False
                    scores[b_i, :-1].masked_fill_(radius_mask_high.to(scores.device), float('-inf'))

                    low_hop_penalty = torch.clamp(r_min - hops_from_curr, min=0.0)
                    low_hop_penalty[goal_idx] = 0.0
                    scores[b_i, :-1] = scores[b_i, :-1] - (
                        low_hop_penalty.to(scores.device)
                        * float(self.decode_bias_cfg.get('radius_low_penalty_scale', 0.75))
                    )

                    # --- 3. soft directional bias ---
                    curr_pos = node_positions[c_idx]  # [2]
                    goal_pos = node_positions[goal_idx] # [2]
                    
                    target_vec = goal_pos - curr_pos
                    target_norm = torch.norm(target_vec)
                    
                    if target_norm > 1e-5: # 목적지에 이미 도달한게 아니라면
                        all_vecs = node_positions - curr_pos # [N, 2]
                        all_norms = torch.norm(all_vecs, dim=1).clamp(min=1e-8)
                        
                        eps = 1e-8
                        cos_sim = (all_vecs[:, 0] * target_vec[0] + all_vecs[:, 1] * target_vec[1]) / (all_norms * (target_norm + eps))
                        corridor_ok = (hops_from_curr + apsp_matrix[:scores.size(1) - 1, goal_idx].float()) <= (current_goal_hops + float(self.decode_bias_cfg.get('corridor_slack', 2.0)))
                        directional_penalty = torch.clamp(
                            float(self.decode_bias_cfg.get('directional_cos_margin', -0.25)) - cos_sim,
                            min=0.0,
                        )
                        directional_penalty = directional_penalty * float(
                            self.decode_bias_cfg.get('directional_penalty_scale', 1.25)
                        )
                        directional_penalty = directional_penalty * torch.where(
                            corridor_ok,
                            torch.full_like(directional_penalty, 0.35),
                            torch.ones_like(directional_penalty),
                        )
                        directional_penalty[goal_idx] = 0.0
                        scores[b_i, :-1] = scores[b_i, :-1] - directional_penalty.to(scores.device)

                    bias_payload = compute_manager_decode_bias(
                        apsp_matrix=apsp_matrix,
                        start_idx=start_idx,
                        current_idx=c_idx,
                        goal_idx=goal_idx,
                        eos_index=EOS_INDEX,
                        cfg=self.decode_bias_cfg,
                        generated_tokens_so_far=k,
                    )
                    scores[b_i] = scores[b_i] + bias_payload['total_bias'].to(scores.device)

            if finished_mask.any():
                scores[finished_mask] = float('-inf')
                scores[finished_mask, EOS_INDEX] = 0.0

            # SL 데이터에는 빈 계획이 없으므로, 첫 토큰에서 즉시 EOS는 허용하지 않는다.
            if k == 0:
                scores[:, EOS_INDEX] = float('-inf')
            
            # Select
            if temperature <= 1e-5:
                next_node_indices = torch.argmax(scores, dim=-1)
            else:
                # NaN/Inf 방어: 대형 그래프에서 score overflow 방지
                scores = torch.clamp(scores, min=-1e6, max=1e6)
                scores[torch.isnan(scores)] = float('-inf')
                probs = F.softmax(scores / temperature, dim=-1)
                probs = torch.clamp(probs, min=0.0)
                probs[torch.isnan(probs)] = 0.0
                # 전체 행이 0이면 uniform으로 fallback
                zero_rows = probs.sum(dim=-1) < 1e-8
                if zero_rows.any():
                    probs[zero_rows] = 1.0 / probs.size(-1)
                next_node_indices = torch.multinomial(probs, 1).squeeze(-1)
            
            full_seqs.append(next_node_indices)
            
            # Check EOS
            is_eos = (next_node_indices == EOS_INDEX)
            
            # Update Visited
            row_indices = torch.arange(batch_size, device=x.device)
            newly_selected_nodes = (~finished_mask) & (~is_eos)
            if newly_selected_nodes.any():
                visited_mask[row_indices[newly_selected_nodes], next_node_indices[newly_selected_nodes]] = True
            
            # Prepare Next Input
            selected_emb = memory_extended[row_indices, next_node_indices, :].unsqueeze(1)
            curr_emb = selected_emb
            finished_mask = finished_mask | is_eos
            
            if finished_mask.all():
                break
            
        full_seqs = torch.stack(full_seqs, dim=1)
        eos_mask = (full_seqs == EOS_INDEX)
        has_eos = eos_mask.any(dim=1)
        if has_eos.any():
            first_eos = eos_mask.float().argmax(dim=1)
            time_idx = torch.arange(full_seqs.size(1), device=x.device).unsqueeze(0)
            pad_after_eos = time_idx > first_eos.unsqueeze(1)
            full_seqs = full_seqs.masked_fill(pad_after_eos & has_eos.unsqueeze(1), self.PAD_TOKEN)
        return full_seqs, None
