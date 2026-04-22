import os
import sys
from datetime import datetime

# [Warning Suppression] Filter PyTorch Prototype Warnings
import warnings
warnings.filterwarnings("ignore", ".*nested.*", category=UserWarning)

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# [Fix #6] 반복 import 방지: 루프 내부에서 호출하던 것을 상단으로 이동
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pad_sequence

from src.models.manager import GraphTransformerManager
from src.models.worker import WorkerLSTM
from src.utils.graph_loader import GraphLoader
from src.data.segment_loader import HierarchicalDataset, hierarchical_collate

print("Imports done. Starting... ", flush=True)

def train_sl(args):
    print(">>> STARTING SL TRAINING <<<", flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_mgr = device
    device_wkr = torch.device('cuda:1') if torch.cuda.device_count() > 1 and args.parallel else device_mgr
    tqdm.write(f"Active Device (Manager): {device_mgr}")
    tqdm.write(f"Active Device (Worker): {device_wkr}")
    
    # MEMORY OPTIMIZATION: Try loading Pre-processed Tensors
    # MEMORY OPTIMIZATION: Enforce Pre-processed Tensors
    # We REMOVE the fallback to pickle to prevent accidental 23GB memory usage.
    
    pt_manager_path = 'data/manager_data.pt'
    pt_worker_path = 'data/worker_data.pt'

    # 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                
    if os.path.exists(pt_manager_path) and os.path.exists(pt_worker_path):
        tqdm.write("✅ Loading Optimized Tensor Data (.pt)...")
        mgr_samples = torch.load(pt_manager_path)
        wkr_samples = torch.load(pt_worker_path)
        
        # Checkpoints no longer needed for Full Path strategy
        # if os.path.exists('data/checkpoints.pkl'): ...
                
    else:
        # Force failure instead of silent fallback
        print("❌ Optimized data not found!", flush=True)
        print("   Please run: python src/data/generate_expert.py --count 50000")
        print("   This will generate 'manager_data.pt' and 'worker_data.pt'.")
        sys.exit(1) # Prevent 23GB memory explosion
    
    loader = GraphLoader(f'data/{args.map}_node.tntp', f'data/{args.map}_net.tntp')
    pyg_data = loader.get_pyg_data()
    num_nodes = pyg_data.x.size(0)
    
    # APSP 행렬 계산: Worker의 거리/방향 피처용
    tqdm.write("🌍 Computing APSP for Worker features...")
    G = loader.get_nx_graph()
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    length_apsp = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
    apsp_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for u_id, lengths in length_apsp.items():
        u_idx = node_to_idx[u_id]
        for v_id, dist_val in lengths.items():
            v_idx = node_to_idx[v_id]
            apsp_matrix[u_idx, v_idx] = dist_val
    max_dist = apsp_matrix.max().item()
    tqdm.write(f"✅ APSP Ready. Max Distance: {max_dist:.2f}")

    tqdm.write("🔗 Computing Hop APSP for Manager masking...")
    hop_apsp = dict(nx.all_pairs_shortest_path_length(G))
    hop_apsp_matrix = torch.full((num_nodes, num_nodes), float('inf'), dtype=torch.float32)
    for u_id, lengths in hop_apsp.items():
        u_idx = node_to_idx[u_id]
        for v_id, hop_val in lengths.items():
            v_idx = node_to_idx[v_id]
            hop_apsp_matrix[u_idx, v_idx] = float(hop_val)
    hop_apsp_device = hop_apsp_matrix.to(device)
    tqdm.write("✅ Hop APSP Ready.")
    
    # [DAgger] APSP Next Hop 테이블 계산: next_hop[src][dst] = src에서 dst로 최단경로의 첫 번째 노드
    # Worker가 틀린 위치에 있을 때 "올바른 다음 행동"을 재계산하는 데 사용
    print("🔄 Computing APSP Next Hop Table for DAgger...", flush=True)
    all_paths = dict(nx.all_pairs_dijkstra_path(G, weight='length'))
    apsp_next_hop = torch.full((num_nodes, num_nodes), -1, dtype=torch.long)
    for src_id, paths in all_paths.items():
        src_idx = node_to_idx[src_id]
        for dst_id, path in paths.items():
            dst_idx = node_to_idx[dst_id]
            if len(path) >= 2:
                # 최단 경로의 두 번째 노드 = 다음 hop
                apsp_next_hop[src_idx, dst_idx] = node_to_idx[path[1]]
            else:
                # src == dst인 경우 자기 자신
                apsp_next_hop[src_idx, dst_idx] = src_idx
    apsp_next_hop_device = apsp_next_hop.to(device)
    del all_paths  # 메모리 해제
    tqdm.write(f"✅ Next Hop Table Ready. Shape: {apsp_next_hop.shape}")
    
    # [Fix #1] GPU 캐싱 먼저 수행 (학습 루프에서 실제로 사용할 텐서)
    apsp_device_global = apsp_matrix.to(device)
    
    mgr_dataset = HierarchicalDataset(mgr_samples, pyg_data, mode='manager')
    
    # Worker에 APSP 행렬 전달 (net_dist, dir_x, dir_y 피처 계산용)
    wkr_dataset = HierarchicalDataset(wkr_samples, pyg_data, mode='worker',
                                       apsp_matrix=apsp_matrix, max_dist=max_dist)
    # [v4] Worker State에 위상학적 특성(hop)을 주입하기 위해 Dataset 객체에 hop_matrix 부착
    wkr_dataset.hop_matrix = hop_apsp_matrix
    
    # [Fix #1] Dataset 생성 완료 후 CPU 원본 해제 (이중 보유 방지)
    del apsp_matrix
    import gc; gc.collect()
    
    # === Train/Validation Split (80/20) ===
    from torch.utils.data import random_split
    
    mgr_train_size = int(0.8 * len(mgr_dataset))
    mgr_val_size = len(mgr_dataset) - mgr_train_size
    mgr_train_ds, mgr_val_ds = random_split(mgr_dataset, [mgr_train_size, mgr_val_size])
    
    wkr_train_size = int(0.8 * len(wkr_dataset))
    wkr_val_size = len(wkr_dataset) - wkr_train_size
    wkr_train_ds, wkr_val_ds = random_split(wkr_dataset, [wkr_train_size, wkr_val_size])
    
    tqdm.write(f"📊 Manager: Train={mgr_train_size}, Val={mgr_val_size}")
    tqdm.write(f"📊 Worker: Train={wkr_train_size}, Val={wkr_val_size}")
    
    # Clean up
    if 'data_dict' in locals():
        del data_dict
    import gc; gc.collect()
    
    # DATA LOADER OMPTIMIZATION:
    num_workers = 0 # Force single process for debug/stability on Windows 
    print(f"Using {num_workers} DataLoader workers (Optimal for Mem/Speed)...")
    if args.parallel:
        mgr_batch_size = int(args.batch_size * 1.5)
        wkr_batch_size = args.batch_size
    else:
        mgr_batch_size = args.batch_size
        wkr_batch_size = args.batch_size
    
    # Train Loaders
    mgr_loader = DataLoader(mgr_train_ds, batch_size=mgr_batch_size, shuffle=True, collate_fn=hierarchical_collate, num_workers=0, pin_memory=True)
    wkr_loader = DataLoader(wkr_train_ds, batch_size=wkr_batch_size, shuffle=True, collate_fn=hierarchical_collate, num_workers=0, pin_memory=True)
    
    # Validation Loaders
    mgr_val_loader = DataLoader(mgr_val_ds, batch_size=mgr_batch_size, shuffle=False, collate_fn=hierarchical_collate, num_workers=0, pin_memory=True)
    wkr_val_loader = DataLoader(wkr_val_ds, batch_size=wkr_batch_size, shuffle=False, collate_fn=hierarchical_collate, num_workers=0, pin_memory=True)
    
    # edge_dim=3: [length, capacity, speed] → 인덱스 [0, 7, 8]
    manager = GraphTransformerManager(
        node_dim=4, 
        hidden_dim=args.hidden_dim, 
        dropout=0.2,
        edge_dim=3
    ).to(device_mgr)
    
    worker = WorkerLSTM(
        node_dim=8,  # [v3] [is_curr, is_tgt, net_dist, dir_x, dir_y, is_final, hop_dist, time_pct]
        hidden_dim=args.hidden_dim,
        edge_dim=3
    ).to(device_wkr)
    
    
    # [Hyperparam] Separate LRs
    optimizer_mgr = optim.Adam(manager.parameters(), lr=args.lr_manager)
    wkr_opt = optim.Adam(worker.parameters(), lr=args.lr_worker)
    
    # Learning Rate Schedulers
    # Manager: ReduceLROnPlateau (Plateau detection)
    mgr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mgr, mode='min', factor=0.5, patience=5)
    
    # Worker: Cosine Annealing (Stable decay) — T_max는 Worker 실제 에포크 수 기준
    wkr_scheduler = optim.lr_scheduler.CosineAnnealingLR(wkr_opt, T_max=max(1, args.epochs // 2), eta_min=args.lr_worker * 0.01)
    
    mgr_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    wkr_criterion = nn.CrossEntropyLoss()
    
    # 타임스탬프 서브폴더 생성: logs/sl_pretrain/<YYYY-MM-DD_HHMM>_sl_ep<N>/
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    run_label = f"{timestamp}_sl_ep{args.epochs}"
    save_dir = os.path.join("logs", "sl_pretrain", run_label)
    os.makedirs(save_dir, exist_ok=True)
    
    # 학습 곡선 기록용
    history = {
        'mgr_train_loss': [], 'mgr_val_loss': [],
        'wkr_train_loss': [], 'wkr_val_loss': [],
    }
    
    # === Early Stopping 설정 ===
    # Worker: 과적합 감지 시 학습 중단 (best 가중치 복원)
    # Manager: 장기간 개선 없으면 전체 학습 종료
    WORKER_ES_PATIENCE = 5   # Worker val loss가 5회 연속 개선 안 되면 학습 중단
    MANAGER_ES_PATIENCE = 7  # Manager val loss가 7회 연속 개선 안 되면 전체 종료
    
    best_wkr_val = float('inf')
    best_mgr_val = float('inf')
    wkr_es_counter = 0  # Worker early stopping 카운터
    mgr_es_counter = 0  # Manager early stopping 카운터
    wkr_frozen = False  # True이면 Worker 학습 건너뜀 (과적합 감지됨)
    
    # Best 가중치 저장 (메모리에 보관)
    import copy
    best_wkr_state = copy.deepcopy(worker.state_dict())
    best_mgr_state = copy.deepcopy(manager.state_dict())
    
    # [Fix #1] APSP 행렬은 이미 GPU에 캐싱 완료 (apsp_device_global)
    
    def run_sequential_pipeline():
        nonlocal best_mgr_val, mgr_es_counter, best_mgr_state, best_wkr_val, wkr_es_counter, best_wkr_state, wkr_frozen
        for epoch in range(1, args.epochs + 1):
            tqdm.write(f"\nEpoch {epoch}/{args.epochs}")
        
            # --- Manager Train ---
            if not getattr(args, 'worker_only', False):
                manager.train()
            mgr_loss = 0
            mgr_batches = 0
            mgr_correct = 0
            mgr_total = 0
        
            pbar = tqdm(mgr_loader, desc="Manager", leave=False, dynamic_ncols=True)
            for batch in pbar:
                batch = batch.to(device)
                # === Manager Training (Inductive Pointer Network) ===
                optimizer_mgr.zero_grad()
            
                # [Robustness] Dynamic Token Definition based on current batch's graph
                # 1. Project Features & Dense Batching
                # [Refactor] Teacher Forcing Logic:
                # We calculate embeddings EXTERNALLY and pass them to Manager.
                mgr_edge_attr = batch.edge_attr[:, [0, 7, 8]] if batch.edge_attr is not None else None
                node_emb_all = manager.topology_enc(batch.x, batch.edge_index, edge_attr=mgr_edge_attr) # [B*N, Hidden]
            
                # [Fix #6] import는 파일 상단으로 이동 완료
                node_emb_dense, mask_dense = to_dense_batch(node_emb_all, batch.batch) # [B, N_max, H]
                x_dense, _ = to_dense_batch(batch.x, batch.batch) # [B, N_max, F]
            
                B, N_max, H = node_emb_dense.size()
            
                # Dynamic Special Tokens
                # Valid Nodes: 0 ~ N_max-1
                # EOS Index: N_max (Virtual Node)
                # PAD Index: -100 (Ignored in Loss)
                EOS_IDX_for_pointer = N_max
            
                # 2. Prepare Target Sequence (Indices)
                raw_targets = batch.y # [B, L]
                is_valid = (raw_targets != -100)
            
                # 3. Create Target Embeddings
                # Extend Reference Embeddings with EOS
                eos_emb_expanded = manager.eos_token_emb.expand(B, 1, H)
                full_ref_embs = torch.cat([node_emb_dense, eos_emb_expanded], dim=1) # [B, N+1, H]
            
                # Gather mask & indices
                # raw_targets has indices 0~N-1, or -100
                safe_gather_idx = raw_targets.clone()
                safe_gather_idx[~is_valid] = 0
            
                # Gather Embeddings [B, L, H]
                target_seq_emb = torch.gather(full_ref_embs, 1, safe_gather_idx.unsqueeze(-1).expand(-1, -1, H))
            
                # Mask out embedding for PAD positions
                target_seq_emb = target_seq_emb * is_valid.unsqueeze(-1).float()
            
                # Call Forward
                # target_seq_emb: [B, L, H]
                # edge_attr: Manager DataLoader에 edge_attr가 없을 수 있으므로 None-safe 처리
                mgr_edge_attr = batch.edge_attr[:, [0, 7, 8]] if batch.edge_attr is not None else None
                pointer_logits = manager(batch.x, batch.edge_index, batch.batch, target_seq_emb=target_seq_emb, edge_attr=mgr_edge_attr)
                # logits: [B, L+1, N+1] (Time steps shifted by 1 due to SOS)
            
                # 3. Construct Final Targets [B, L+1]
                # Input:  [SOS, n1, n2, ..., nk] (implicitly constructed by fwd)
                # Target: [n1,  n2, ..., nk, EOS]
            
                seq_L = raw_targets.size(1)
                final_targets = torch.full((B, seq_L + 1), -100, dtype=torch.long, device=device)
            
                lengths = is_valid.sum(dim=1) # [B]
            
                # [Optimize] 파이썬 루핑 제거 및 벡터화 연산 도입 (CPU-GPU 동기화 병목 해소)
                valid_mask = torch.arange(seq_L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                final_targets[:, :seq_L][valid_mask] = raw_targets[valid_mask]
            
                batch_indices = torch.arange(B, device=device)
                final_targets[batch_indices, lengths] = EOS_IDX_for_pointer
            
                # Flatten for Loss
                logits_flat = pointer_logits.view(-1, N_max + 1) # [B*(L+1), N+1]
                targets_flat = final_targets.view(-1)            # [B*(L+1)]
            
                # 1. NLL Loss
                nll_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
            
                # 2. Soft Label Loss (Only for Real Node Targets)
                # Mask for Real Nodes (Target < N_max and Target != -100)
                real_node_mask = (targets_flat < N_max) & (targets_flat >= 0)
            
                if real_node_mask.sum() > 0:
                    # [Fix 5] EOS 토큰을 제외한 순수 공간 노드에 대해서만 log_softmax 적용
                    logits_nodes = pointer_logits[..., :-1] # [B, L+1, N_max]
                    log_probs_nodes = F.log_softmax(logits_nodes, dim=-1) 
                
                    # Get Network Distances for Soft Label
                    # We use the pre-computed APSP matrix for topologically accurate distances.
                    real_mask_2d = (final_targets < N_max) & (final_targets >= 0)
                    safe_targets_2d = final_targets.clone()
                    safe_targets_2d[~real_mask_2d] = 0
                
                    dists = apsp_device_global[safe_targets_2d] # [B, L+1, N_max]
                
                    # [Hyperparam] Soft Label Temperature
                    temperature = 1.0 
                    soft_probs = F.softmax(-dists / temperature, dim=-1) # [B, L+1, N_max]
                    del dists  # [Fix #2] Soft Label 중간 텐서 즉시 해제 (~50MB/batch 회수)
                
                    kl_loss_raw = F.kl_div(log_probs_nodes, soft_probs, reduction='none', log_target=False) # [B, L+1, N_max]
                
                    # Sum over node classes, then mask
                    kl_loss = (kl_loss_raw.sum(dim=-1) * real_mask_2d.float()).sum() / (real_mask_2d.sum() + 1e-6)
                else:
                    kl_loss = torch.tensor(0.0, device=device)
            
                # Total Loss
                loss = 0.5 * nll_loss + 0.5 * kl_loss
            
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf Loss detected. Skipping batch.")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(manager.parameters(), max_norm=1.0)
                optimizer_mgr.step()
            
                mgr_loss += loss.item()
                mgr_batches += 1
            
                # Accuracy
                pred_tokens = torch.argmax(pointer_logits, dim=-1)
                correct = (pred_tokens == final_targets) & (final_targets != -100)
                mgr_correct += correct.sum().item()
                mgr_total += (final_targets != -100).sum().item()
            
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*mgr_correct/max(1,mgr_total):.1f}%"})
            
            tqdm.write(f"Manager Loss: {mgr_loss/max(1, mgr_batches):.4f}")
            
            # --- Worker Train ---
            # [Improved] Worker 학습 빈도 2배 증가 (20에폭에서 10회)
            WORKER_FREQ = 2
        
            # [DAgger] TF Ratio 감소: 0.3까지 감소 (원래 경로 감독 유지)
            # 0.0으로 내리면 CE/KL 충돌로 성능 하락 → 70% 자율 + 30% Teacher로 균형
            tf_ratio = max(0.3, 1.0 - 1.5 * (epoch - 1) / args.epochs)
            window_size = 5 # 다중 스텝 언롤링(Multi-step Unrolling) 윈도우 크기
        
            if WORKER_FREQ > 0 and epoch % WORKER_FREQ == 0 and not wkr_frozen:
                worker.train()
                wkr_loss = 0
                wkr_batches = 0
                wkr_correct = 0
                wkr_total_steps = 0
            
                pbar = tqdm(wkr_loader, desc=f"Worker (Ep {epoch})", leave=False, dynamic_ncols=True)
                for batch in pbar:
                    batch = batch.to(device)
                
                    # 시퀀스 데이터 로드
                    # segment_loader에서 c_nodes, t_nodes, n_hops 리스트 반환함
                    c_seqs = batch.c_nodes
                    t_seqs = batch.t_nodes
                    n_seqs = batch.n_hops
                
                    num_graphs = len(c_seqs) # 배치 내 그래프(경로) 개수
                
                    wkr_opt.zero_grad()
                
                    # [Optimization] 단일 텐서 기반 은닉 상태 관리 (리스트 제거)
                    h = torch.zeros(num_graphs, args.hidden_dim, device=device)
                    c = torch.zeros(num_graphs, args.hidden_dim, device=device)
                
                    # [Optimization] 벡터 추출을 위한 시퀀스 텐서 패딩 (CPU-GPU 동기화 병목 방지)
                    # [Fix #6] pad_sequence import는 파일 상단으로 이동 완료
                    c_seqs_pad = pad_sequence(c_seqs, batch_first=True, padding_value=0) # [B, Max_Seq_Len]
                       # [v4] Global Z-score 정규화: GATv2 GraphNorm 성능 극대화 및 공변량 편이 방지
                    _ea_mean = base_edge_attr.mean(dim=0, keepdim=True)
                    _ea_std = base_edge_attr.std(dim=0, keepdim=True).clamp(min=1e-8)
                    base_edge_attr = (base_edge_attr - _ea_mean) / _ea_std         
                    max_seq_len = c_seqs_pad.size(1)
                
                    loss_buffer = [] # 윈도우 손실 버퍼
                
                    # 이전 스텝의 예측 노드 (Autoregressive 용도)
                    prev_preds = c_seqs_pad[:, 0].clone() # [B]
                
                    # [Optimization v3] 동적 배치 축소를 위한 기본 변수 사전 정의 (루프 밖)
                    node_coords_per_graph = batch.x[:, :2].view(num_graphs, -1, 2)  # [B, N, 2]
                    num_nodes_per_graph = node_coords_per_graph.size(1)
                    base_edges = pyg_data.edge_index.to(device)  # [2, E_single]
                    base_edge_attr = pyg_data.edge_attr[:, [0, 7, 8]].to(device)  # [E_single, 3] length+capacity+speed
                    # [v3] Min-Max 정규화: GATv2 attention 편향 방지
                    _ea_min = base_edge_attr.min(dim=0, keepdim=True)[0]
                    _ea_max = base_edge_attr.max(dim=0, keepdim=True)[0]
                    base_edge_attr = (base_edge_attr - _ea_min) / (_ea_max - _ea_min).clamp(min=1e-8)
                    num_edges_per_graph = base_edges.size(1)
                
                    # [Fix A] Pre-allocation: 초기 K=num_graphs에 대해 사전 계산
                    # K가 줄어들 때만 재계산 (대부분의 스텝에서 재활용)
                    cached_K = num_graphs
                    cached_offsets = torch.arange(num_graphs, device=device) * num_nodes_per_graph
                    cached_edge_offset = cached_offsets.view(num_graphs, 1, 1)
                    cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                    cached_edge_attr = base_edge_attr.repeat(num_graphs, 1)  # [E_single*K, edge_dim]
                    cached_batch_vec = torch.arange(num_graphs, device=device).repeat_interleave(num_nodes_per_graph)
                
                    for step in range(max_seq_len):
                        # === 1. 활성 그래프 인덱스 도출 ===
                        active_mask = step < seq_lens  # [B] 전체 배치 마스크
                        if not active_mask.any(): break
                    
                        active_idx = active_mask.nonzero(as_tuple=True)[0]  # [K] 활성 인덱스
                        K = active_idx.size(0)
                    
                        # [Fix A] K가 변했을 때만 edge_index/batch_vec 재계산
                        # (edge_index/batch_vec는 gradient graph에 참여하지 않으므로 캐싱 안전)
                        if K != cached_K:
                            cached_K = K
                            cached_offsets = torch.arange(K, device=device) * num_nodes_per_graph
                            cached_edge_offset = cached_offsets.view(K, 1, 1)
                            cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                            cached_edge_attr = base_edge_attr.repeat(K, 1)  # [E_single*K, edge_dim]
                            cached_batch_vec = torch.arange(K, device=device).repeat_interleave(num_nodes_per_graph)
                    
                        # === 2. 패딩된 텐서에서 활성 그래프만 슬라이싱 ===
                        act_step = torch.clamp(torch.tensor(step, device=device), max=seq_lens[active_idx] - 1)
                    
                        use_tf = torch.rand(K, device=device) < tf_ratio
                        curr_nodes_k = torch.where(
                            use_tf | (step == 0),
                            c_seqs_pad[active_idx, step],
                            prev_preds[active_idx]
                        )
                        tgt_nodes_k = t_seqs_pad[active_idx, act_step]
                    
                        # [DAgger] 정답 재계산: 모델 예측 위치에 있을 때 APSP 기반 올바른 행동
                        # Teacher Forcing이 아닌 경우 → 현재 위치에서 서브골까지의 
                        # [Fix 4] Subset 우회
                        ds = wkr_loader.dataset.dataset if hasattr(wkr_loader.dataset, 'dataset') else wkr_loader.dataset
                    
                        original_labels = n_seqs_pad[active_idx, act_step]  # 원래 정답
                        dagger_labels = apsp_next_hop_device[curr_nodes_k, tgt_nodes_k]  # APSP 재계산 정답
                    
                        # TF 사용 시 원래 정답, 아닌 경우 DAgger 정답 사용
                        # 단, DAgger 정답이 -1이면 (경로 없음) 원래 정답 유지
                        valid_dagger = (dagger_labels >= 0)
                        target_labels_k = torch.where(
                            (use_tf | (step == 0)) | (~valid_dagger),
                            original_labels,
                            dagger_labels
                        )  # [K]
                    
                        # === 3. 컴팩트 GNN 입력 조립 ===
                        # gradient checkpoint는 backward 시 입력을 재사용하므로
                        # in-place 수정 대신 매 스텝 새 텐서 생성 (CUDA 캐싱 풀이 크기 동일 시 즉시 재할당)
                        compact_coords = node_coords_per_graph[active_idx].reshape(K * num_nodes_per_graph, 2)
                    
                        worker_in = torch.zeros((K * num_nodes_per_graph, 8), device=device)
                        worker_in[cached_offsets + curr_nodes_k, 0] = 1.0  # is_current
                        worker_in[cached_offsets + tgt_nodes_k, 1] = 1.0   # is_target
                    
                        # APSP 거리 피처
                        if hasattr(ds, 'apsp_matrix'):
                            max_dst = ds.max_dist
                            active_dists = apsp_device_global[tgt_nodes_k] / max_dst  # [K, N]
                            worker_in[:, 2] = active_dists.reshape(-1)
                        else:
                            worker_in[:, 2] = 0.0
                    
                        # 방향 벡터 피처
                        tgt_coords = compact_coords[cached_offsets + tgt_nodes_k]  # [K, 2]
                        tgt_coords_rep = tgt_coords.repeat_interleave(num_nodes_per_graph, dim=0)  # [K*N, 2]
                        diff = tgt_coords_rep - compact_coords
                        norm_val = diff.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        worker_in[:, 3:5] = diff / norm_val
                        is_final_target = (act_step == (seq_lens[active_idx] - 1)).float()
                        worker_in[:, 5] = is_final_target.repeat_interleave(num_nodes_per_graph)

                        # [Track 1] hop_dist 피처 (col 8): 서브골까지의 정규화된 홉 거리
                        if hasattr(ds, 'hop_matrix'):
                            if not hasattr(ds, '_cached_max_h'):
                                _hm = ds.hop_matrix
                                ds._cached_max_h = float(_hm[_hm < float('inf')].max().item()) if _hm.numel() > 0 else 50.0
                                ds._cached_hm_dev = _hm.to(device)
                            hop_dists_k = ds._cached_hm_dev[curr_nodes_k, tgt_nodes_k]
                            worker_in[:, 6] = torch.clamp(hop_dists_k.float() / ds._cached_max_h, 0.0, 1.0).reshape(-1)
                        else:
                            worker_in[:, 6] = 0.0
                    
                        # === 4. 활성 은닉 상태 슬라이싱 & Forward ===
                        h_active = h[active_idx]  # [K, H]
                        c_active = c[active_idx]  # [K, H]
                    
                        scores, h_next_k, c_next_k, _ = worker.predict_next_hop(
                            worker_in, cached_edge_index, h_active, c_active, cached_batch_vec,
                            edge_attr=cached_edge_attr
                        )
                    
                        # === 5. 은닉 상태 원본 텐서에 매핑 (BPTT 유지) ===
                        h_new = h.clone()
                        c_new = c.clone()
                        h_new[active_idx] = h_next_k
                        c_new[active_idx] = c_next_k
                        h = h_new
                        c = c_new
                    
                        # === 6. 손실 계산: Heuristic-Guided SL (CE + KL) ===
                        scores_view = scores.view(K, num_nodes_per_graph)
                    
                        # CE 손실: 정답 노드 학습 (DAgger 시 재계산된 정답 사용)
                        ce_loss = F.cross_entropy(scores_view, target_labels_k)
                    
                        # KL 손실: APSP 기반 Soft Label
                        # [Fix] DAgger 시 curr_nodes_k 기준 서브골까지 거리 사용 (CE와 일치)
                        # TF 모드: 원래 서브골(tgt_nodes_k) 기준
                        # DAgger 모드: 동일하게 서브골(tgt_nodes_k) 기준이지만 curr 위치가 다름
                        dist_to_target = apsp_device_global[:, tgt_nodes_k].T / max_dist  # [K, N]
                        wkr_soft_labels = F.softmax(-dist_to_target / 0.1, dim=-1)  # T=0.1 (sharp)
                        log_probs = F.log_softmax(scores_view, dim=-1)
                        kl_loss = F.kl_div(log_probs, wkr_soft_labels, reduction='batchmean', log_target=False)
                    
                        # [Fix] DAgger 비율에 따라 KL 가중치 조절
                        # TF 비율이 낮을수록 DAgger 정답과 KL이 충돌할 수 있으므로 KL 가중치 감소
                        kl_weight = 0.3 * tf_ratio + 0.1 * (1.0 - tf_ratio)  # TF=1→0.3, TF=0.3→0.21
                        step_loss = (1.0 - kl_weight) * ce_loss + kl_weight * kl_loss
                    
                        # Autoregressive: 예측 노드 갱신
                        pred_k = scores_view.argmax(dim=1)  # [K]
                        prev_preds = prev_preds.clone()
                        prev_preds[active_idx] = pred_k
                            
                        loss_buffer.append(step_loss)
                    
                        # Accuracy 추적: 활성 K개 그래프의 정답 여부
                        wkr_correct += (pred_k == target_labels_k).sum().item()
                        wkr_total_steps += K
                    
                        if (step + 1) % window_size == 0 or (step + 1) == max_seq_len:
                            if len(loss_buffer) > 0:
                                avg_window_loss = sum(loss_buffer) / len(loss_buffer)
                                avg_window_loss.backward()
                                wkr_loss += avg_window_loss.item()
                                loss_buffer.clear()
                        
                            # 단일 텐서 BPTT Detach
                            h = h.detach()
                            c = c.detach()
                        
                            torch.nn.utils.clip_grad_norm_(worker.parameters(), max_norm=1.0)
                            wkr_opt.step()
                            wkr_opt.zero_grad()
                        
                        if not active_mask.any():
                            loss_buffer.clear()
                        
                        # 메모리 해제
                        del worker_in, scores, h_next_k, c_next_k, h_new, c_new, scores_view, step_loss
                        
                    wkr_batches += (max_seq_len / window_size) # Approximate batches
                    wkr_acc = 100 * wkr_correct / max(1, wkr_total_steps)
                    pbar.set_postfix({'loss': f"{wkr_loss/max(1, wkr_batches):.4f}", 'acc': f"{wkr_acc:.1f}%"})
            
                tqdm.write(f"Worker Train Loss: {wkr_loss/max(1, wkr_batches):.4f}, Acc: {wkr_acc:.1f}% (TF Ratio: {tf_ratio:.2f})")
            elif wkr_frozen:
                pass
            else:
                pass
        
            # === Validation Loss ===
            manager.eval()
            worker.eval()
            mgr_val_loss = 0
            wkr_val_loss = 0
            mgr_val_batches = 0
            wkr_val_batches = 0
        
            with torch.no_grad():
                # Manager Validation
                for batch in mgr_val_loader:
                    batch = batch.to(device)
                
                    # [Robustness] Dynamic Token Definition
                    # [Refactor] Validation Embedding Gathering
                    mgr_edge_attr = batch.edge_attr[:, [0, 7, 8]] if batch.edge_attr is not None else None
                    node_emb_all = manager.topology_enc(batch.x, batch.edge_index, edge_attr=mgr_edge_attr)
                
                    # [Fix #6] import는 파일 상단으로 이동 완료
                    node_emb_dense, mask_dense = to_dense_batch(node_emb_all, batch.batch) # [B, N_max, H]
                    x_dense, _ = to_dense_batch(batch.x, batch.batch) # Needed for Soft Label
                    B, N_max, H = node_emb_dense.size()
                
                    EOS_IDX_for_pointer = N_max
                    PAD_VAL = -100
                
                    raw_targets = batch.y # [B, L]
                    is_valid = (raw_targets != PAD_VAL)
                
                    # 1. Construct Target Embeddings with EOS extension
                    eos_emb_expanded = manager.eos_token_emb.expand(B, 1, H)
                    full_ref_embs = torch.cat([node_emb_dense, eos_emb_expanded], dim=1) # [B, N+1, H]
                
                    # Gather mask & indices
                    safe_gather_idx = raw_targets.clone()
                    safe_gather_idx[~is_valid] = 0
                
                    target_seq_emb = torch.gather(full_ref_embs, 1, safe_gather_idx.unsqueeze(-1).expand(-1, -1, H))
                    target_seq_emb = target_seq_emb * is_valid.unsqueeze(-1).float()
                
                    # 2. Forward Pass
                    val_edge_attr = batch.edge_attr[:, [0, 7, 8]] if batch.edge_attr is not None else None
                    pointer_logits = manager(batch.x, batch.edge_index, batch.batch, target_seq_emb=target_seq_emb, edge_attr=val_edge_attr)
                
                    # 3. Construct Targets for Loss
                    seq_L = raw_targets.size(1)
                    final_targets = torch.full((B, seq_L + 1), PAD_VAL, dtype=torch.long, device=device)
                
                    lengths = is_valid.sum(dim=1)
                
                    # [Optimize] 파이썬 루핑 제거 및 벡터화 연산 도입
                    valid_mask = torch.arange(seq_L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                    final_targets[:, :seq_L][valid_mask] = raw_targets[valid_mask]
                
                    batch_indices = torch.arange(B, device=device)
                    final_targets[batch_indices, lengths] = EOS_IDX_for_pointer

                    # Flatten for Loss
                    logits_flat = pointer_logits.view(-1, N_max + 1)
                    targets_flat = final_targets.view(-1)
                
                    # 4. Hybrid Loss Calculation
                    nll_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_VAL)
                
                    # Soft Label Logic
                    real_node_mask = (targets_flat < N_max) & (targets_flat >= 0)
                    if real_node_mask.sum() > 0:
                        # [Fix 5] EOS 토큰을 제외한 순수 공간 노드에 대해서만 log_softmax 적용
                        logits_nodes = pointer_logits[..., :-1]
                        log_probs_nodes = F.log_softmax(logits_nodes, dim=-1)
                    
                        real_mask_2d = (final_targets < N_max) & (final_targets >= 0)
                        safe_targets_2d = final_targets.clone()
                        safe_targets_2d[~real_mask_2d] = 0
                    
                        dists = apsp_device_global[safe_targets_2d]
                    
                        temperature = 1.0
                        soft_probs = F.softmax(-dists / temperature, dim=-1)
                        del dists  # [Fix #2] Validation Soft Label 중간 텐서 즉시 해제
                        kl_loss_raw = F.kl_div(log_probs_nodes, soft_probs, reduction='none', log_target=False)
                        kl_loss = (kl_loss_raw.sum(dim=-1) * real_mask_2d.float()).sum() / (real_mask_2d.sum() + 1e-6)
                    else:
                        kl_loss = torch.tensor(0.0, device=device)
                    
                    loss = 0.5 * nll_loss + 0.5 * kl_loss
                    mgr_val_loss += loss.item()
                    mgr_val_batches += 1
            
                # Worker Validation (Frequency: 1/5)
                if epoch % WORKER_FREQ == 0:
                    for batch in wkr_val_loader:
                        batch = batch.to(device)
                    
                        c_seqs = batch.c_nodes
                        t_seqs = batch.t_nodes
                        n_seqs = batch.n_hops
                    
                        val_num_graphs = len(c_seqs)
                    
                        # [Optimization v3] 단일 텐서 기반 은닉 상태 (Validation)
                        h = torch.zeros(val_num_graphs, args.hidden_dim, device=device)
                        c = torch.zeros(val_num_graphs, args.hidden_dim, device=device)
                    
                        # [Optimization] Validation 텐서 패딩 (CPU-GPU 싱크 병목 차단)
                        # [Fix #6] pad_sequence import는 파일 상단으로 이동 완료
                        c_seqs_pad = pad_sequence(c_seqs, batch_first=True, padding_value=0)
                        t_seqs_pad = pad_sequence(t_seqs, batch_first=True, padding_value=0)
                        n_seqs_pad = pad_sequence(n_seqs, batch_first=True, padding_value=0)
                        seq_lens = torch.tensor([seq.size(0) for seq in c_seqs], dtype=torch.long, device=device)
                    
                        max_seq_len = c_seqs_pad.size(1)
                    
                        # [Optimization v3] 동적 배치 축소용 사전 정의 (Validation)
                        node_coords_per_graph = batch.x[:, :2].view(val_num_graphs, -1, 2)  # [B, N, 2]
                        num_nodes_per_graph = node_coords_per_graph.size(1)
                        base_edges = pyg_data.edge_index.to(device)
                        base_edge_attr = pyg_data.edge_attr[:, [0, 7, 8]].to(device)  # length+capacity+speed
                        # [v3] Min-Max 정규화
                        _ea_min = base_edge_attr.min(dim=0, keepdim=True)[0]
                        _ea_max = base_edge_attr.max(dim=0, keepdim=True)[0]
                        base_edge_attr = (base_edge_attr - _ea_min) / (_ea_max - _ea_min).clamp(min=1e-8)
                    
                        # [Fix A] Pre-allocation (Validation) — edge_index/batch_vec만 캐싱
                        cached_K = val_num_graphs
                        cached_offsets = torch.arange(val_num_graphs, device=device) * num_nodes_per_graph
                        cached_edge_offset = cached_offsets.view(val_num_graphs, 1, 1)
                        cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                        cached_edge_attr = base_edge_attr.repeat(val_num_graphs, 1)
                        cached_batch_vec = torch.arange(val_num_graphs, device=device).repeat_interleave(num_nodes_per_graph)
                    
                        # [Fix 4] Subset 우회
                        val_ds = wkr_val_loader.dataset.dataset if hasattr(wkr_val_loader.dataset, 'dataset') else wkr_val_loader.dataset
                    
                        for step in range(max_seq_len):
                            # === 1. 활성 그래프 인덱스 도출 ===
                            active_mask = step < seq_lens
                            if not active_mask.any(): break
                        
                            active_idx = active_mask.nonzero(as_tuple=True)[0]  # [K]
                            K = active_idx.size(0)
                        
                            # [Fix A] K 변경 시에만 재계산
                            if K != cached_K:
                                cached_K = K
                                cached_offsets = torch.arange(K, device=device) * num_nodes_per_graph
                                cached_edge_offset = cached_offsets.view(K, 1, 1)
                                cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                                cached_edge_attr = base_edge_attr.repeat(K, 1)
                                cached_batch_vec = torch.arange(K, device=device).repeat_interleave(num_nodes_per_graph)
                        
                            # === 2. 활성 그래프만 슬라이싱 ===
                            act_step = torch.clamp(torch.tensor(step, device=device), max=seq_lens[active_idx] - 1)
                        
                            curr_nodes_k = c_seqs_pad[active_idx, step]
                            tgt_nodes_k = t_seqs_pad[active_idx, act_step]
                            target_labels_k = n_seqs_pad[active_idx, act_step]
                        
                            # === 3. 컴팩트 GNN 입력 (새 텐서 생성) ===
                            compact_coords = node_coords_per_graph[active_idx].reshape(K * num_nodes_per_graph, 2)
                        
                            worker_in = torch.zeros((K * num_nodes_per_graph, 8), device=device)
                            worker_in[cached_offsets + curr_nodes_k, 0] = 1.0
                            worker_in[cached_offsets + tgt_nodes_k, 1] = 1.0
                        
                            # APSP 거리 피처
                            val_ds = wkr_val_loader.dataset.dataset if hasattr(wkr_val_loader.dataset, 'dataset') else wkr_val_loader.dataset
                            if hasattr(val_ds, 'apsp_matrix'):
                                max_dst = val_ds.max_dist
                                active_dists = apsp_device_global[tgt_nodes_k] / max_dst
                                worker_in[:, 2] = active_dists.reshape(-1)
                            else:
                                worker_in[:, 2] = 0.0
                        
                            # 방향 벡터 피처
                            tgt_coords = compact_coords[cached_offsets + tgt_nodes_k]
                            tgt_coords_rep = tgt_coords.repeat_interleave(num_nodes_per_graph, dim=0)
                            diff = tgt_coords_rep - compact_coords
                            norm_val = diff.norm(dim=1, keepdim=True).clamp(min=1e-8)
                            worker_in[:, 3:5] = diff / norm_val
                            is_final_target = (act_step == (seq_lens[active_idx] - 1)).float()
                            worker_in[:, 5] = is_final_target.repeat_interleave(num_nodes_per_graph)

                            # [Track 1] hop_dist 피처 (col 8)
                            if hasattr(val_ds, 'hop_matrix'):
                                if not hasattr(val_ds, '_cached_max_h'):
                                    _hm = val_ds.hop_matrix
                                    val_ds._cached_max_h = float(_hm[_hm < float('inf')].max().item()) if _hm.numel() > 0 else 50.0
                                    val_ds._cached_hm_dev = _hm.to(device)
                                hop_dists_k = val_ds._cached_hm_dev[curr_nodes_k, tgt_nodes_k]
                                worker_in[:, 6] = torch.clamp(hop_dists_k.float() / val_ds._cached_max_h, 0.0, 1.0).reshape(-1)
                            else:
                                worker_in[:, 6] = 0.0
                        
                            # === 4. Forward ===
                            h_active = h[active_idx]
                            c_active = c[active_idx]
                        
                            scores, h_next_k, c_next_k, _ = worker.predict_next_hop(
                                worker_in, cached_edge_index, h_active, c_active, cached_batch_vec,
                                edge_attr=cached_edge_attr
                            )
                        
                            # [Fix #3] no_grad() 내부이므로 clone 불필요 → 직접 대입
                            h[active_idx] = h_next_k
                            c[active_idx] = c_next_k
                        
                            # === 6. 손실 계산: Heuristic-Guided SL (Validation) ===
                            scores_view = scores.view(K, num_nodes_per_graph)
                            ce_loss = F.cross_entropy(scores_view, target_labels_k)
                            dist_to_target = apsp_device_global[:, tgt_nodes_k].T / max_dist
                            wkr_soft_labels = F.softmax(-dist_to_target / 0.1, dim=-1)
                            log_probs = F.log_softmax(scores_view, dim=-1)
                            kl_loss = F.kl_div(log_probs, wkr_soft_labels, reduction='batchmean', log_target=False)
                            step_loss = 0.7 * ce_loss + 0.3 * kl_loss
                            
                            wkr_val_loss += step_loss.item()
                            wkr_val_batches += 1
            
                    # [Fix #4] 유효 변수만 해제 (h_list, c_list는 이전 코드 잔해)
                    pass
        
            mgr_val = mgr_val_loss/max(1, mgr_val_batches)
            if epoch % WORKER_FREQ == 0 and not wkr_frozen:
                wkr_val = wkr_val_loss/max(1, wkr_val_batches)
                tqdm.write(f"📊 Validation - Manager: {mgr_val:.4f}, Worker: {wkr_val:.4f}")
                # wkr_val logging
                history['wkr_train_loss'].append(wkr_loss/max(1, wkr_batches))
                history['wkr_val_loss'].append(wkr_val)
            
                # === Worker Early Stopping 체크 ===
                if wkr_val < best_wkr_val:
                    best_wkr_val = wkr_val
                    wkr_es_counter = 0
                    best_wkr_state = copy.deepcopy(worker.state_dict())
                    tqdm.write(f"  ✅ Worker Best Val Loss 갱신: {best_wkr_val:.4f}")
                else:
                    wkr_es_counter += 1
                    tqdm.write(f"  ⚠️ Worker Val Loss 미개선 ({wkr_es_counter}/{WORKER_ES_PATIENCE})")
                    if wkr_es_counter >= WORKER_ES_PATIENCE:
                        tqdm.write(f"  ⛔ Worker Early Stopping! Best Val={best_wkr_val:.4f}에서 동결.")
                        tqdm.write(f"     → Worker 가중치를 Best 시점으로 복원합니다.")
                        worker.load_state_dict(best_wkr_state)
                        wkr_frozen = True
            elif wkr_frozen:
                pass
            else:
                pass
        
            # === Manager Early Stopping 체크 ===
            if mgr_val < best_mgr_val:
                best_mgr_val = mgr_val
                mgr_es_counter = 0
                best_mgr_state = copy.deepcopy(manager.state_dict())
                tqdm.write(f"  ✅ Manager Best Val Loss 갱신: {best_mgr_val:.4f}")
            else:
                mgr_es_counter += 1
                tqdm.write(f"  ⚠️ Manager Val Loss 미개선 ({mgr_es_counter}/{MANAGER_ES_PATIENCE})")
        
            # 학습 곡선 기록
            mgr_train = mgr_loss/max(1, mgr_batches)
            # wkr_train handled above
            history['mgr_train_loss'].append(mgr_train)
            history['mgr_val_loss'].append(mgr_val)
        
            manager.train()
            if not wkr_frozen:
                worker.train()
        
            # Step LR Schedulers
            mgr_scheduler.step(mgr_val) # ReduceLROnPlateau needs metric
            if epoch % WORKER_FREQ == 0 and not wkr_frozen:
                wkr_scheduler.step()
        
            # Get LR safely
            current_lr_mgr = optimizer_mgr.param_groups[0]['lr']
            current_lr_wkr = wkr_opt.param_groups[0]['lr']
            tqdm.write(f"Current LR - Manager: {current_lr_mgr:.6f}, Worker: {current_lr_wkr:.6f}")

            # === Manager Early Stopping: 전체 학습 종료 ===
            if mgr_es_counter >= MANAGER_ES_PATIENCE:
                tqdm.write(f"\n⛔ Manager Early Stopping! Best Val={best_mgr_val:.4f}")
                print(f"   → 에폭 {epoch}에서 학습 조기 종료.")
                manager.load_state_dict(best_mgr_state)
                if not wkr_frozen:
                    # Worker도 마지막 best로 복원
                    worker.load_state_dict(best_wkr_state)
                break
    


    # Worker GPU allocation copies
    apsp_device_wkr = apsp_device_global.to(device_wkr) if 'apsp_device_global' in locals() else None
    apsp_next_hop_wkr = apsp_next_hop_device.to(device_wkr) if 'apsp_next_hop_device' in locals() else None
    
    def run_manager_pipeline():
        nonlocal best_mgr_val, mgr_es_counter, best_mgr_state
        mgr_pbar = tqdm(total=args.epochs, desc="Manager SL", position=0, dynamic_ncols=True)
        for epoch in range(1, args.epochs + 1):
            # --- Manager Train ---
            if not getattr(args, 'worker_only', False):
                manager.train()
            mgr_loss = 0
            mgr_batches = 0
            mgr_correct = 0
            mgr_total = 0
        
            mgr_batch_iter = tqdm(mgr_loader, desc=f"  Mgr Train Ep{epoch}", leave=False, position=2, dynamic_ncols=True)
            for batch in mgr_batch_iter:
                batch = batch.to(device)
                # === Manager Training (Inductive Pointer Network) ===
                optimizer_mgr.zero_grad()
            
                # [Robustness] Dynamic Token Definition based on current batch's graph
                # 1. Project Features & Dense Batching
                # [Refactor] Teacher Forcing Logic:
                # We calculate embeddings EXTERNALLY and pass them to Manager.
                mgr_edge_attr = batch.edge_attr[:, [0, 7, 8]] if batch.edge_attr is not None else None
                node_emb_all = manager.topology_enc(batch.x, batch.edge_index, edge_attr=mgr_edge_attr) # [B*N, Hidden]
            
                # [Fix #6] import는 파일 상단으로 이동 완료
                node_emb_dense, mask_dense = to_dense_batch(node_emb_all, batch.batch) # [B, N_max, H]
                x_dense, _ = to_dense_batch(batch.x, batch.batch) # [B, N_max, F]
            
                B, N_max, H = node_emb_dense.size()
            
                # Dynamic Special Tokens
                # Valid Nodes: 0 ~ N_max-1
                # EOS Index: N_max (Virtual Node)
                # PAD Index: -100 (Ignored in Loss)
                EOS_IDX_for_pointer = N_max
            
                # 2. Prepare Target Sequence (Indices)
                raw_targets = batch.y # [B, L]
                is_valid = (raw_targets != -100)
            
                # 3. Create Target Embeddings
                # Extend Reference Embeddings with EOS
                eos_emb_expanded = manager.eos_token_emb.expand(B, 1, H)
                full_ref_embs = torch.cat([node_emb_dense, eos_emb_expanded], dim=1) # [B, N+1, H]
            
                # Gather mask & indices
                # raw_targets has indices 0~N-1, or -100
                safe_gather_idx = raw_targets.clone()
                safe_gather_idx[~is_valid] = 0
            
                # Gather Embeddings [B, L, H]
                target_seq_emb = torch.gather(full_ref_embs, 1, safe_gather_idx.unsqueeze(-1).expand(-1, -1, H))
            
                # Mask out embedding for PAD positions
                target_seq_emb = target_seq_emb * is_valid.unsqueeze(-1).float()
            
                # Call Forward
                # target_seq_emb: [B, L, H]
                # edge_attr: Manager DataLoader에 edge_attr가 없을 수 있으므로 None-safe 처리
                mgr_edge_attr = batch.edge_attr[:, [0, 7, 8]] if batch.edge_attr is not None else None
                pointer_logits = manager(batch.x, batch.edge_index, batch.batch, target_seq_emb=target_seq_emb, edge_attr=mgr_edge_attr)
                # logits: [B, L+1, N+1] (Time steps shifted by 1 due to SOS)
            
                # 3. Construct Final Targets [B, L+1]
                # Input:  [SOS, n1, n2, ..., nk] (implicitly constructed by fwd)
                # Target: [n1,  n2, ..., nk, EOS]
            
                seq_L = raw_targets.size(1)
                final_targets = torch.full((B, seq_L + 1), -100, dtype=torch.long, device=device)
            
                lengths = is_valid.sum(dim=1) # [B]
            
                # [Optimize] 파이썬 루핑 제거 및 벡터화 연산 도입 (CPU-GPU 동기화 병목 해소)
                valid_mask = torch.arange(seq_L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                final_targets[:, :seq_L][valid_mask] = raw_targets[valid_mask]
            
                batch_indices = torch.arange(B, device=device)
                final_targets[batch_indices, lengths] = EOS_IDX_for_pointer
            
                # Flatten for Loss
                logits_flat = pointer_logits.view(-1, N_max + 1) # [B*(L+1), N+1]
                targets_flat = final_targets.view(-1)            # [B*(L+1)]
            
                # 1. NLL Loss
                nll_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
            
                # 2. Soft Label Loss (Only for Real Node Targets)
                # Mask for Real Nodes (Target < N_max and Target != -100)
                real_node_mask = (targets_flat < N_max) & (targets_flat >= 0)
            
                if real_node_mask.sum() > 0:
                    # [Fix 5] EOS 토큰을 제외한 순수 공간 노드에 대해서만 log_softmax 적용
                    logits_nodes = pointer_logits[..., :-1] # [B, L+1, N_max]
                    log_probs_nodes = F.log_softmax(logits_nodes, dim=-1) 
                
                    # Get Network Distances for Soft Label
                    # We use the pre-computed APSP matrix for topologically accurate distances.
                    real_mask_2d = (final_targets < N_max) & (final_targets >= 0)
                    safe_targets_2d = final_targets.clone()
                    safe_targets_2d[~real_mask_2d] = 0
                
                    dists = apsp_device_global[safe_targets_2d] # [B, L+1, N_max]
                
                    # [Hyperparam] Soft Label Temperature
                    temperature = 1.0 
                    soft_probs = F.softmax(-dists / temperature, dim=-1) # [B, L+1, N_max]
                    del dists  # [Fix #2] Soft Label 중간 텐서 즉시 해제 (~50MB/batch 회수)
                
                    kl_loss_raw = F.kl_div(log_probs_nodes, soft_probs, reduction='none', log_target=False) # [B, L+1, N_max]
                
                    # Sum over node classes, then mask
                    kl_loss = (kl_loss_raw.sum(dim=-1) * real_mask_2d.float()).sum() / (real_mask_2d.sum() + 1e-6)
                else:
                    kl_loss = torch.tensor(0.0, device=device)
            
                # Total Loss
                loss = 0.5 * nll_loss + 0.5 * kl_loss
            
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf Loss detected. Skipping batch.")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(manager.parameters(), max_norm=1.0)
                optimizer_mgr.step()
            
                mgr_loss += loss.item()
                mgr_batches += 1
            
                # Accuracy
                pred_tokens = torch.argmax(pointer_logits, dim=-1)
                correct = (pred_tokens == final_targets) & (final_targets != -100)
                mgr_correct += correct.sum().item()
                mgr_total += (final_targets != -100).sum().item()
                
                mgr_batch_iter.set_postfix({
                    'loss': f"{mgr_loss/max(1, mgr_batches):.4f}", 
                    'lr': f"{optimizer_mgr.param_groups[0]['lr']:.2e}", 
                    'acc': f"{100*mgr_correct/max(1,mgr_total):.1f}%"
                })
            
            mgr_pbar.set_postfix({
                'loss': f"{mgr_loss/max(1, mgr_batches):.4f}", 
                'lr': f"{optimizer_mgr.param_groups[0]['lr']:.2e}", 
                'acc': f"{100*mgr_correct/max(1,mgr_total):.1f}%"
            })
            
            manager.eval()
            mgr_val_loss = 0
            mgr_val_batches = 0
            with torch.no_grad():
                # Manager Validation
                mgr_val_iter = tqdm(mgr_val_loader, desc=f"  Mgr Val Ep{epoch}", leave=False, position=2, dynamic_ncols=True)
                for batch in mgr_val_iter:
                    batch = batch.to(device)
                
                    # [Robustness] Dynamic Token Definition
                    # [Refactor] Validation Embedding Gathering
                    mgr_edge_attr = batch.edge_attr[:, [0, 7, 8]] if batch.edge_attr is not None else None
                    node_emb_all = manager.topology_enc(batch.x, batch.edge_index, edge_attr=mgr_edge_attr)
                
                    # [Fix #6] import는 파일 상단으로 이동 완료
                    node_emb_dense, mask_dense = to_dense_batch(node_emb_all, batch.batch) # [B, N_max, H]
                    x_dense, _ = to_dense_batch(batch.x, batch.batch) # Needed for Soft Label
                    B, N_max, H = node_emb_dense.size()
                
                    EOS_IDX_for_pointer = N_max
                    PAD_VAL = -100
                
                    raw_targets = batch.y # [B, L]
                    is_valid = (raw_targets != PAD_VAL)
                
                    # 1. Construct Target Embeddings with EOS extension
                    eos_emb_expanded = manager.eos_token_emb.expand(B, 1, H)
                    full_ref_embs = torch.cat([node_emb_dense, eos_emb_expanded], dim=1) # [B, N+1, H]
                
                    # Gather mask & indices
                    safe_gather_idx = raw_targets.clone()
                    safe_gather_idx[~is_valid] = 0
                
                    target_seq_emb = torch.gather(full_ref_embs, 1, safe_gather_idx.unsqueeze(-1).expand(-1, -1, H))
                    target_seq_emb = target_seq_emb * is_valid.unsqueeze(-1).float()
                
                    # 2. Forward Pass
                    val_edge_attr = batch.edge_attr[:, [0, 7, 8]] if batch.edge_attr is not None else None
                    pointer_logits = manager(batch.x, batch.edge_index, batch.batch, target_seq_emb=target_seq_emb, edge_attr=val_edge_attr)
                
                    # 3. Construct Targets for Loss
                    seq_L = raw_targets.size(1)
                    final_targets = torch.full((B, seq_L + 1), PAD_VAL, dtype=torch.long, device=device)
                
                    lengths = is_valid.sum(dim=1)
                
                    # [Optimize] 파이썬 루핑 제거 및 벡터화 연산 도입
                    valid_mask = torch.arange(seq_L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                    final_targets[:, :seq_L][valid_mask] = raw_targets[valid_mask]
                
                    batch_indices = torch.arange(B, device=device)
                    final_targets[batch_indices, lengths] = EOS_IDX_for_pointer

                    # Flatten for Loss
                    logits_flat = pointer_logits.view(-1, N_max + 1)
                    targets_flat = final_targets.view(-1)
                
                    # 4. Hybrid Loss Calculation
                    nll_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_VAL)
                
                    # Soft Label Logic
                    real_node_mask = (targets_flat < N_max) & (targets_flat >= 0)
                    if real_node_mask.sum() > 0:
                        # [Fix 5] EOS 토큰을 제외한 순수 공간 노드에 대해서만 log_softmax 적용
                        logits_nodes = pointer_logits[..., :-1]
                        log_probs_nodes = F.log_softmax(logits_nodes, dim=-1)
                    
                        real_mask_2d = (final_targets < N_max) & (final_targets >= 0)
                        safe_targets_2d = final_targets.clone()
                        safe_targets_2d[~real_mask_2d] = 0
                    
                        dists = apsp_device_global[safe_targets_2d]
                    
                        temperature = 1.0
                        soft_probs = F.softmax(-dists / temperature, dim=-1)
                        del dists  # [Fix #2] Validation Soft Label 중간 텐서 즉시 해제
                        kl_loss_raw = F.kl_div(log_probs_nodes, soft_probs, reduction='none', log_target=False)
                        kl_loss = (kl_loss_raw.sum(dim=-1) * real_mask_2d.float()).sum() / (real_mask_2d.sum() + 1e-6)
                    else:
                        kl_loss = torch.tensor(0.0, device=device)
                    
                    loss = 0.5 * nll_loss + 0.5 * kl_loss
                    mgr_val_loss += loss.item()
                    mgr_val_batches += 1
                    
                    mgr_val_iter.set_postfix({'loss': f"{mgr_val_loss/max(1, mgr_val_batches):.4f}"})
            
            mgr_val = mgr_val_loss/max(1, mgr_val_batches)
            # === Manager Early Stopping 체크 ===
            if mgr_val < best_mgr_val:
                best_mgr_val = mgr_val
                mgr_es_counter = 0
                best_mgr_state = copy.deepcopy(manager.state_dict())
                tqdm.write(f"  ✅ Manager Best Val Loss 갱신: {best_mgr_val:.4f}")
            else:
                mgr_es_counter += 1
                tqdm.write(f"  ⚠️ Manager Val Loss 미개선 ({mgr_es_counter}/{MANAGER_ES_PATIENCE})")
        
            mgr_train = mgr_loss/max(1, mgr_batches)
            history['mgr_train_loss'].append(mgr_train)
            history['mgr_val_loss'].append(mgr_val)
            mgr_scheduler.step(mgr_val) # ReduceLROnPlateau needs metric
            current_lr_mgr = optimizer_mgr.param_groups[0]['lr']
            mgr_pbar.set_postfix({'loss': f"{mgr_val:.4f}", 'lr': f"{current_lr_mgr:.2e}"})
            mgr_pbar.update(1)
            # === Manager Early Stopping: 전체 학습 종료 ===
            if mgr_es_counter >= MANAGER_ES_PATIENCE:
                tqdm.write(f"\n⛔ Manager Early Stopping! Best Val={best_mgr_val:.4f}")
                print(f"   → 에폭 {epoch}에서 학습 조기 종료.", flush=True)
                manager.load_state_dict(best_mgr_state)
                break

    def run_worker_pipeline():
        nonlocal best_wkr_val, wkr_es_counter, best_wkr_state, wkr_frozen
        WORKER_FREQ = 2  # Worker 에포크 = args.epochs // WORKER_FREQ
        wkr_epochs = args.epochs // WORKER_FREQ
        wkr_pbar = tqdm(total=wkr_epochs, desc="Worker SL", position=1, dynamic_ncols=True)
        for epoch in range(1, wkr_epochs + 1):
            # --- Worker Train ---
            # [v3] Parallel 모드: Worker 전용 에포크 수만큼 매 에포크 학습
        
            # [DAgger] TF Ratio 감소: 0.3까지 감소 (원래 경로 감독 유지)
            tf_ratio = max(0.3, 1.0 - 1.5 * (epoch - 1) / wkr_epochs)
            window_size = 5  # 다중 스텝 언롤링(Multi-step Unrolling) 윈도우 크기
        
            if not wkr_frozen:
                worker.train()
                wkr_loss = 0
                wkr_batches = 0
                wkr_correct = 0
                wkr_total_steps = 0
            
                wkr_batch_iter = tqdm(wkr_loader, desc=f"  Worker Ep{epoch}", leave=False, position=2, dynamic_ncols=True)
                batch_idx = 0
                for batch in wkr_batch_iter:
                    batch = batch.to(device_wkr)
                
                    # 시퀀스 데이터 로드
                    # segment_loader에서 c_nodes, t_nodes, n_hops 리스트 반환함
                    c_seqs = batch.c_nodes
                    t_seqs = batch.t_nodes
                    n_seqs = batch.n_hops
                
                    num_graphs = len(c_seqs) # 배치 내 그래프(경로) 개수
                
                    wkr_opt.zero_grad()
                
                    # [Optimization] 단일 텐서 기반 은닉 상태 관리 (리스트 제거)
                    h = torch.zeros(num_graphs, args.hidden_dim, device=device_wkr)
                    c = torch.zeros(num_graphs, args.hidden_dim, device=device_wkr)
                
                    # [Optimization] 벡터 추출을 위한 시퀀스 텐서 패딩 (CPU-GPU 동기화 병목 방지)
                    # [Fix #6] pad_sequence import는 파일 상단으로 이동 완료
                    c_seqs_pad = pad_sequence(c_seqs, batch_first=True, padding_value=0) # [B, Max_Seq_Len]
                    t_seqs_pad = pad_sequence(t_seqs, batch_first=True, padding_value=0)
                    n_seqs_pad = pad_sequence(n_seqs, batch_first=True, padding_value=0)
                    seq_lens = torch.tensor([seq.size(0) for seq in c_seqs], dtype=torch.long, device=device_wkr)
                
                    max_seq_len = c_seqs_pad.size(1)
                
                    loss_buffer = [] # 윈도우 손실 버퍼
                
                    # 이전 스텝의 예측 노드 (Autoregressive 용도)
                    prev_preds = c_seqs_pad[:, 0].clone() # [B]
                
                    # [Optimization v3] 동적 배치 축소를 위한 기본 변수 사전 정의 (루프 밖)
                    node_coords_per_graph = batch.x[:, :2].view(num_graphs, -1, 2)  # [B, N, 2]
                    num_nodes_per_graph = node_coords_per_graph.size(1)
                    base_edges = pyg_data.edge_index.to(device_wkr)  # [2, E_single]
                    base_edge_attr = pyg_data.edge_attr[:, [0, 7, 8]].to(device_wkr)  # [E_single, 3] length+capacity+speed
                    # [v3] Min-Max 정규화: GATv2 attention 편향 방지
                    _ea_min = base_edge_attr.min(dim=0, keepdim=True)[0]
                    _ea_max = base_edge_attr.max(dim=0, keepdim=True)[0]
                    base_edge_attr = (base_edge_attr - _ea_min) / (_ea_max - _ea_min).clamp(min=1e-8)
                    num_edges_per_graph = base_edges.size(1)
                
                    # [Fix A] Pre-allocation: 초기 K=num_graphs에 대해 사전 계산
                    # K가 줄어들 때만 재계산 (대부분의 스텝에서 재활용)
                    cached_K = num_graphs
                    cached_offsets = torch.arange(num_graphs, device=device_wkr) * num_nodes_per_graph
                    cached_edge_offset = cached_offsets.view(num_graphs, 1, 1)
                    cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                    cached_edge_attr = base_edge_attr.repeat(num_graphs, 1)  # [E_single*K, edge_dim]
                    cached_batch_vec = torch.arange(num_graphs, device=device_wkr).repeat_interleave(num_nodes_per_graph)
                
                    for step in range(max_seq_len):
                        # === 1. 활성 그래프 인덱스 도출 ===
                        active_mask = step < seq_lens  # [B] 전체 배치 마스크
                        if not active_mask.any(): break
                    
                        active_idx = active_mask.nonzero(as_tuple=True)[0]  # [K] 활성 인덱스
                        K = active_idx.size(0)
                    
                        # [Fix A] K가 변했을 때만 edge_index/batch_vec 재계산
                        # (edge_index/batch_vec는 gradient graph에 참여하지 않으므로 캐싱 안전)
                        if K != cached_K:
                            cached_K = K
                            cached_offsets = torch.arange(K, device=device_wkr) * num_nodes_per_graph
                            cached_edge_offset = cached_offsets.view(K, 1, 1)
                            cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                            cached_edge_attr = base_edge_attr.repeat(K, 1)  # [E_single*K, edge_dim]
                            cached_batch_vec = torch.arange(K, device=device_wkr).repeat_interleave(num_nodes_per_graph)
                    
                        # === 2. 패딩된 텐서에서 활성 그래프만 슬라이싱 ===
                        act_step = torch.clamp(torch.tensor(step, device=device_wkr), max=seq_lens[active_idx] - 1)
                    
                        use_tf = torch.rand(K, device=device_wkr) < tf_ratio
                        curr_nodes_k = torch.where(
                            use_tf | (step == 0),
                            c_seqs_pad[active_idx, step],
                            prev_preds[active_idx]
                        )
                        tgt_nodes_k = t_seqs_pad[active_idx, act_step]
                    
                        # [DAgger] 정답 재계산: 모델 예측 위치에 있을 때 APSP 기반 올바른 행동
                        # Teacher Forcing이 아닌 경우 → 현재 위치에서 서브골까지의 
                        # [Fix 4] Subset 우회
                        ds = wkr_loader.dataset.dataset if hasattr(wkr_loader.dataset, 'dataset') else wkr_loader.dataset
                    
                        original_labels = n_seqs_pad[active_idx, act_step]  # 원래 정답
                        dagger_labels = apsp_next_hop_wkr[curr_nodes_k, tgt_nodes_k]  # APSP 재계산 정답
                    
                        # TF 사용 시 원래 정답, 아닌 경우 DAgger 정답 사용
                        # 단, DAgger 정답이 -1이면 (경로 없음) 원래 정답 유지
                        valid_dagger = (dagger_labels >= 0)
                        target_labels_k = torch.where(
                            (use_tf | (step == 0)) | (~valid_dagger),
                            original_labels,
                            dagger_labels
                        )  # [K]
                    
                        # === 3. 컴팩트 GNN 입력 조립 (v4 Architecture) ===
                        compact_coords = node_coords_per_graph[active_idx].reshape(K * num_nodes_per_graph, 2)
                        
                        worker_in = torch.zeros((K * num_nodes_per_graph, 8), device=device_wkr)
                        
                        # 0: is_curr
                        worker_in[cached_offsets + curr_nodes_k, 0] = 1.0  
                        
                        # 1: is_subgoal
                        worker_in[cached_offsets + tgt_nodes_k, 1] = 1.0   
                        
                        # SL 훈련을 위한 최종 목적지 추출 (시퀀스의 마지막 노드)
                        final_nodes_k = c_seqs_pad[active_idx, seq_lens[active_idx] - 1]
                        
                        # 2: is_final_goal
                        worker_in[cached_offsets + final_nodes_k, 2] = 1.0 
                        
                        # 위상학적 특성 캐싱 확인
                        if not hasattr(ds, '_cached_max_h_wkr'):
                            _hm = ds.hop_matrix
                            ds._cached_max_h_wkr = float(_hm[_hm < float('inf')].max().item()) if _hm.numel() > 0 else 50.0
                            ds._cached_hm_dev_wkr = _hm.to(device_wkr)
                        
                        # 3: hop_to_subgoal (각 노드 -> 서브골)
                        hop_to_sub_k = ds._cached_hm_dev_wkr[:, tgt_nodes_k].T  # [K, N]
                        worker_in[:, 3] = torch.clamp(hop_to_sub_k.float() / ds._cached_max_h_wkr, 0.0, 1.0).reshape(-1)
                        
                        # 4: hop_to_final (각 노드 -> 최종 목적지)
                        hop_to_final_k = ds._cached_hm_dev_wkr[:, final_nodes_k].T  # [K, N]
                        worker_in[:, 4] = torch.clamp(hop_to_final_k.float() / ds._cached_max_h_wkr, 0.0, 1.0).reshape(-1)
                        
                        # 5, 6: global_heading_x, y (각 노드 -> 최종 목적지 유클리드 방향 단위벡터)
                        final_coords_k = compact_coords[cached_offsets + final_nodes_k]  # [K, 2]
                        final_coords_rep = final_coords_k.repeat_interleave(num_nodes_per_graph, dim=0)  # [K*N, 2]
                        diff_final = final_coords_rep - compact_coords
                        norm_final = diff_final.norm(dim=1, keepdim=True).clamp(min=1e-6)
                        worker_in[:, 5:7] = diff_final / norm_final
                        
                        # 7: time_to_go (남은 스텝 비율)
                        time_to_go_val = 1.0 - (float(step) / max(max_seq_len, 1))
                        worker_in[:, 7] = time_to_go_val
                    
                        # === 4. 활성 은닉 상태 슬라이싱 & Forward ===
                        h_active = h[active_idx]  # [K, H]
                        c_active = c[active_idx]  # [K, H]
                    
                        scores, h_next_k, c_next_k, _ = worker.predict_next_hop(
                            worker_in, cached_edge_index, h_active, c_active, cached_batch_vec,
                            edge_attr=cached_edge_attr
                        )
                    
                        # === 5. 은닉 상태 원본 텐서에 매핑 (BPTT 유지) ===
                        h_new = h.clone()
                        c_new = c.clone()
                        h_new[active_idx] = h_next_k
                        c_new[active_idx] = c_next_k
                        h = h_new
                        c = c_new
                    
                        # === 6. 손실 계산: Heuristic-Guided SL (CE + KL) ===
                        scores_view = scores.view(K, num_nodes_per_graph)
                    
                        # CE 손실: 정답 노드 학습 (DAgger 시 재계산된 정답 사용)
                        ce_loss = F.cross_entropy(scores_view, target_labels_k)
                    
                        # KL 손실: APSP 기반 Soft Label
                        # [Fix] DAgger 시 curr_nodes_k 기준 서브골까지 거리 사용 (CE와 일치)
                        # TF 모드: 원래 서브골(tgt_nodes_k) 기준
                        # DAgger 모드: 동일하게 서브골(tgt_nodes_k) 기준이지만 curr 위치가 다름
                        dist_to_target = apsp_device_wkr[:, tgt_nodes_k].T / max_dist  # [K, N]
                        wkr_soft_labels = F.softmax(-dist_to_target / 0.1, dim=-1)  # T=0.1 (sharp)
                        log_probs = F.log_softmax(scores_view, dim=-1)
                        kl_loss = F.kl_div(log_probs, wkr_soft_labels, reduction='batchmean', log_target=False)
                    
                        # [Fix] DAgger 비율에 따라 KL 가중치 조절
                        # TF 비율이 낮을수록 DAgger 정답과 KL이 충돌할 수 있으므로 KL 가중치 감소
                        kl_weight = 0.3 * tf_ratio + 0.1 * (1.0 - tf_ratio)  # TF=1→0.3, TF=0.3→0.21
                        step_loss = (1.0 - kl_weight) * ce_loss + kl_weight * kl_loss
                    
                        # Autoregressive: 예측 노드 갱신
                        pred_k = scores_view.argmax(dim=1)  # [K]
                        prev_preds = prev_preds.clone()
                        prev_preds[active_idx] = pred_k
                            
                        loss_buffer.append(step_loss)
                    
                        # Accuracy 추적: 활성 K개 그래프의 정답 여부
                        wkr_correct += (pred_k == target_labels_k).sum().item()
                        wkr_total_steps += K
                    
                        if (step + 1) % window_size == 0 or (step + 1) == max_seq_len:
                            if len(loss_buffer) > 0:
                                avg_window_loss = sum(loss_buffer) / len(loss_buffer)
                                avg_window_loss.backward()
                                wkr_loss += avg_window_loss.item()
                                loss_buffer.clear()
                        
                            # 단일 텐서 BPTT Detach
                            h = h.detach()
                            c = c.detach()
                        
                            torch.nn.utils.clip_grad_norm_(worker.parameters(), max_norm=1.0)
                            wkr_opt.step()
                            wkr_opt.zero_grad()
                        
                        if not active_mask.any():
                            loss_buffer.clear()
                        
                        # 메모리 해제
                        del worker_in, scores, h_next_k, c_next_k, h_new, c_new, scores_view, step_loss
                        
                    wkr_acc = 100 * wkr_correct / max(1, wkr_total_steps)
                    batch_idx += 1
                    wkr_batch_iter.set_postfix({
                        'loss': f"{wkr_loss/max(1, wkr_batches):.4f}",
                        'acc': f"{wkr_acc:.1f}%",
                        'seq': f"{max_seq_len}"
                    })
                wkr_pbar.set_postfix({
                    'loss': f"{wkr_loss/max(1, wkr_batches):.4f}",
                    'lr': f"{wkr_opt.param_groups[0]['lr']:.2e}",
                    'acc': f"{wkr_acc:.1f}%"
                })
            
                pass
            elif wkr_frozen:
                pass
            else:
                pass
        
            worker.eval()
            wkr_val_loss = 0
            wkr_val_batches = 0
            with torch.no_grad():
                # Worker Validation (매 에포크)
                if not wkr_frozen:
                    wkr_val_iter = tqdm(wkr_val_loader, desc=f"  Worker Val Ep{epoch}", leave=False, position=2, dynamic_ncols=True)
                    for batch in wkr_val_iter:
                        batch = batch.to(device_wkr)
                    
                        c_seqs = batch.c_nodes
                        t_seqs = batch.t_nodes
                        n_seqs = batch.n_hops
                    
                        val_num_graphs = len(c_seqs)
                    
                        # [Optimization v3] 단일 텐서 기반 은닉 상태 (Validation)
                        h = torch.zeros(val_num_graphs, args.hidden_dim, device=device_wkr)
                        c = torch.zeros(val_num_graphs, args.hidden_dim, device=device_wkr)
                    
                        # [Optimization] Validation 텐서 패딩 (CPU-GPU 싱크 병목 차단)
                        # [Fix #6] pad_sequence import는 파일 상단으로 이동 완료
                        c_seqs_pad = pad_sequence(c_seqs, batch_first=True, padding_value=0)
                        t_seqs_pad = pad_sequence(t_seqs, batch_first=True, padding_value=0)
                        n_seqs_pad = pad_sequence(n_seqs, batch_first=True, padding_value=0)
                        seq_lens = torch.tensor([seq.size(0) for seq in c_seqs], dtype=torch.long, device=device_wkr)
                    
                        max_seq_len = c_seqs_pad.size(1)
                    
                        # [Optimization v3] 동적 배치 축소용 사전 정의 (Validation)
                        node_coords_per_graph = batch.x[:, :2].view(val_num_graphs, -1, 2)  # [B, N, 2]
                        num_nodes_per_graph = node_coords_per_graph.size(1)
                        base_edges = pyg_data.edge_index.to(device_wkr)
                        base_edge_attr = pyg_data.edge_attr[:, [0, 7, 8]].to(device_wkr)  # length+capacity+speed
                        # [v4] Global Z-score 정규화
                        _ea_mean = base_edge_attr.mean(dim=0, keepdim=True)
                        _ea_std = base_edge_attr.std(dim=0, keepdim=True).clamp(min=1e-8)
                        base_edge_attr = (base_edge_attr - _ea_mean) / _ea_std
                    
                        # [Fix A] Pre-allocation (Validation) — edge_index/batch_vec만 캐싱
                        cached_K = val_num_graphs
                        cached_offsets = torch.arange(val_num_graphs, device=device_wkr) * num_nodes_per_graph
                        cached_edge_offset = cached_offsets.view(val_num_graphs, 1, 1)
                        cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                        cached_edge_attr = base_edge_attr.repeat(val_num_graphs, 1)
                        cached_batch_vec = torch.arange(val_num_graphs, device=device_wkr).repeat_interleave(num_nodes_per_graph)
                    
                        # [Fix 4] Subset 우회
                        val_ds = wkr_val_loader.dataset.dataset if hasattr(wkr_val_loader.dataset, 'dataset') else wkr_val_loader.dataset
                    
                        for step in range(max_seq_len):
                            # === 1. 활성 그래프 인덱스 도출 ===
                            active_mask = step < seq_lens
                            if not active_mask.any(): break
                        
                            active_idx = active_mask.nonzero(as_tuple=True)[0]  # [K]
                            K = active_idx.size(0)
                        
                            # [Fix A] K 변경 시에만 재계산
                            if K != cached_K:
                                cached_K = K
                                cached_offsets = torch.arange(K, device=device_wkr) * num_nodes_per_graph
                                cached_edge_offset = cached_offsets.view(K, 1, 1)
                                cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                                cached_edge_attr = base_edge_attr.repeat(K, 1)
                                cached_batch_vec = torch.arange(K, device=device_wkr).repeat_interleave(num_nodes_per_graph)
                        
                            # === 2. 활성 그래프만 슬라이싱 ===
                            act_step = torch.clamp(torch.tensor(step, device=device_wkr), max=seq_lens[active_idx] - 1)
                        
                            curr_nodes_k = c_seqs_pad[active_idx, step]
                            tgt_nodes_k = t_seqs_pad[active_idx, act_step]
                            target_labels_k = n_seqs_pad[active_idx, act_step]
                        
                            # === 3. 컴팩트 GNN 입력 (v4 Architecture) ===
                            compact_coords = node_coords_per_graph[active_idx].reshape(K * num_nodes_per_graph, 2)
                        
                            worker_in = torch.zeros((K * num_nodes_per_graph, 8), device=device_wkr)
                            worker_in[cached_offsets + curr_nodes_k, 0] = 1.0
                            worker_in[cached_offsets + tgt_nodes_k, 1] = 1.0
                            
                            # SL Validation을 위한 최종 목적지 추출
                            final_nodes_k = c_seqs_pad[active_idx, seq_lens[active_idx] - 1]
                            worker_in[cached_offsets + final_nodes_k, 2] = 1.0
                        
                            val_ds = wkr_val_loader.dataset.dataset if hasattr(wkr_val_loader.dataset, 'dataset') else wkr_val_loader.dataset
                            
                            # 위상학적 특성 캐싱 확인
                            if not hasattr(val_ds, '_cached_max_h_wkr'):
                                _hm = val_ds.hop_matrix
                                val_ds._cached_max_h_wkr = float(_hm[_hm < float('inf')].max().item()) if _hm.numel() > 0 else 50.0
                                val_ds._cached_hm_dev_wkr = _hm.to(device_wkr)
                                
                            # 3: hop_to_subgoal (각 노드 -> 서브골)
                            hop_to_sub_k = val_ds._cached_hm_dev_wkr[:, tgt_nodes_k].T
                            worker_in[:, 3] = torch.clamp(hop_to_sub_k.float() / val_ds._cached_max_h_wkr, 0.0, 1.0).reshape(-1)
                            
                            # 4: hop_to_final (각 노드 -> 최종 목적지)
                            hop_to_final_k = val_ds._cached_hm_dev_wkr[:, final_nodes_k].T
                            worker_in[:, 4] = torch.clamp(hop_to_final_k.float() / val_ds._cached_max_h_wkr, 0.0, 1.0).reshape(-1)
                            
                            # 5, 6: global_heading_x, y
                            final_coords_k = compact_coords[cached_offsets + final_nodes_k]
                            final_coords_rep = final_coords_k.repeat_interleave(num_nodes_per_graph, dim=0)
                            diff_final = final_coords_rep - compact_coords
                            norm_final = diff_final.norm(dim=1, keepdim=True).clamp(min=1e-6)
                            worker_in[:, 5:7] = diff_final / norm_final
                            
                            # 7: time_to_go
                            time_to_go_val = 1.0 - (float(step) / max(max_seq_len, 1))
                            worker_in[:, 7] = time_to_go_val
                        
                            # === 4. Forward ===
                            h_active = h[active_idx]
                            c_active = c[active_idx]
                        
                            scores, h_next_k, c_next_k, _ = worker.predict_next_hop(
                                worker_in, cached_edge_index, h_active, c_active, cached_batch_vec,
                                edge_attr=cached_edge_attr
                            )
                        
                            # [Fix #3] no_grad() 내부이므로 clone 불필요 → 직접 대입
                            h[active_idx] = h_next_k
                            c[active_idx] = c_next_k
                        
                            # === 6. 손실 계산: Heuristic-Guided SL (Validation) ===
                            scores_view = scores.view(K, num_nodes_per_graph)
                            ce_loss = F.cross_entropy(scores_view, target_labels_k)
                            dist_to_target = apsp_device_wkr[:, tgt_nodes_k].T / max_dist
                            wkr_soft_labels = F.softmax(-dist_to_target / 0.1, dim=-1)
                            log_probs = F.log_softmax(scores_view, dim=-1)
                            kl_loss = F.kl_div(log_probs, wkr_soft_labels, reduction='batchmean', log_target=False)
                            step_loss = 0.7 * ce_loss + 0.3 * kl_loss
                            
                            wkr_val_loss += step_loss.item()
                            wkr_val_batches += 1
                            
                        wkr_val_iter.set_postfix({'loss': f"{wkr_val_loss/max(1, wkr_val_batches):.4f}"})
            
                    # [Fix #4] 유효 변수만 해제 (h_list, c_list는 이전 코드 잔해)
                    pass
        
            if not wkr_frozen:
                wkr_val = wkr_val_loss/max(1, wkr_val_batches)
                tqdm.write(f"📊 Validation - Worker: {wkr_val:.4f}")
                # wkr_val logging
                history['wkr_train_loss'].append(wkr_loss/max(1, wkr_batches))
                history['wkr_val_loss'].append(wkr_val)
            
                # === Worker Early Stopping 체크 ===
                if wkr_val < best_wkr_val:
                    best_wkr_val = wkr_val
                    wkr_es_counter = 0
                    best_wkr_state = copy.deepcopy(worker.state_dict())
                    tqdm.write(f"  ✅ Worker Best Val Loss 갱신: {best_wkr_val:.4f}")
                else:
                    wkr_es_counter += 1
                    tqdm.write(f"  ⚠️ Worker Val Loss 미개선 ({wkr_es_counter}/{WORKER_ES_PATIENCE})")
                    if wkr_es_counter >= WORKER_ES_PATIENCE:
                        tqdm.write(f"  ⛔ Worker Early Stopping! Best Val={best_wkr_val:.4f}에서 동결.")
                        tqdm.write(f"     → Worker 가중치를 Best 시점으로 복원합니다.")
                        worker.load_state_dict(best_wkr_state)
                        wkr_frozen = True
            elif wkr_frozen:
                pass
            else:
                pass
        
            if not wkr_frozen:
                wkr_scheduler.step()
        
            current_lr_wkr = wkr_opt.param_groups[0]['lr']
            wkr_pbar.set_postfix({'loss': f"{best_wkr_val if wkr_frozen else wkr_val_loss/max(1, wkr_val_batches):.4f}", 'lr': f"{current_lr_wkr:.2e}"})
            wkr_pbar.update(1)
            if wkr_frozen:
                tqdm.write(" Worker frozen, stopping thread.")
                break


    if args.parallel:
        print(f"\n🚀 Running in PARALLEL mode with {args.epochs} epochs", flush=True)
        import threading
        t1 = threading.Thread(target=run_manager_pipeline)
        t2 = threading.Thread(target=run_worker_pipeline)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        print("\n=== Parallel Training Finished ===", flush=True)
    else:
        print(f"\n🐢 Running in SEQUENTIAL mode with {args.epochs} epochs", flush=True)
        run_sequential_pipeline()
        print("\n=== Sequential Training Finished ===", flush=True)

    # === Accuracy Evaluation ===
    tqdm.write("\n📊 Evaluating Model Accuracy...")
    manager.eval()
    worker.eval()
    
    # Manager Accuracy: 첫 번째 checkpoint가 A* 경로 위에 있는지
    mgr_exact = 0
    mgr_near = 0  # 1-hop 이내
    mgr_total = 0
    
    # Worker Accuracy: 예측한 next_hop이 정답인지
    wkr_correct = 0
    wkr_total = 0
    
    with torch.no_grad():
        # Manager Eval (샘플 1000개)
        eval_count = min(1000, len(mgr_dataset))
        for i in range(eval_count):
            sample = mgr_dataset[i]
            x_in, edge_index, y, dist_map = sample
            
            x_in = x_in.unsqueeze(0).to(device) if x_in.dim() == 2 else x_in.to(device)
            edge_index = edge_index.to(device)
            y = y.to(device)
            
            # 배치 벡터 생성
            batch_vec = torch.zeros(x_in.size(1) if x_in.dim() == 3 else x_in.size(0), dtype=torch.long, device=device)
            
            # Manager 예측 (첫 번째 토큰만 비교)
            # target_seq_emb=None으로 주면 SOS만 사용하여 첫 토큰 예측
            logits = manager(x_in.squeeze(0), edge_index, batch_vec, target_seq_emb=None, edge_attr=pyg_data.edge_attr[:, [0, 7, 8]].to(device))
            pred = logits[0, 0].argmax().item()
            target = y[0].item() if y.dim() > 0 else y.item()
            
            if pred == target:
                mgr_exact += 1
            
            # distance_map으로 near 체크
            if dist_map is not None:
                if pred < dist_map.size(0) and dist_map[pred].item() <= 1:
                    mgr_near += 1
            
            mgr_total += 1
        
        # Worker Eval (샘플 2000개 - 이제는 시퀀스 경로 개수임)
        eval_count = min(1000, len(wkr_dataset))
        for i in range(eval_count):
            sample = wkr_dataset[i]
            x_raw, edge_index, (c_seq, t_seq, n_seq) = sample
            
            x_raw = x_raw.to(device)
            edge_index = edge_index.to(device)
            c_seq = c_seq.to(device)
            t_seq = t_seq.to(device)
            n_seq = n_seq.to(device)
            
            seq_len = c_seq.size(0)
            h = torch.zeros(1, args.hidden_dim, device=device)
            c = torch.zeros(1, args.hidden_dim, device=device)
            batch_vec = torch.zeros(x_raw.size(0), dtype=torch.long, device=device)
            
            node_coords = x_raw[:, :2]
            num_nodes = x_raw.size(0)
            
            # Autoregressive evaluation
            # [v3] edge_attr 정규화: 루프 밖에서 1회만 수행
            _eval_ea = pyg_data.edge_attr[:, [0, 7, 8]].to(device)
            _ea_min = _eval_ea.min(dim=0, keepdim=True)[0]
            _ea_max = _eval_ea.max(dim=0, keepdim=True)[0]
            _eval_ea = (_eval_ea - _ea_min) / (_ea_max - _ea_min).clamp(min=1e-8)
            
            curr_node_idx = c_seq[0].item()
            
            for step in range(seq_len):
                tgt_node_idx = t_seq[step].item()
                target_label = n_seq[step].item()
                
                flags = torch.zeros(num_nodes, 2, device=device)
                flags[curr_node_idx, 0] = 1.0
                flags[tgt_node_idx, 1] = 1.0
                
                if hasattr(wkr_dataset, 'apsp_matrix'):
                    dist_map = wkr_dataset.apsp_matrix[tgt_node_idx].to(device)
                    net_dist = (dist_map / wkr_dataset.max_dist).unsqueeze(1)
                else:
                    net_dist = torch.zeros(num_nodes, 1, device=device)
                    
                target_pos = node_coords[tgt_node_idx].unsqueeze(0)
                direction = target_pos - node_coords
                norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-8)
                direction = direction / norm
                
                # is_final_target_phase: 1.0 if this step's subgoal is the final target
                is_final = torch.full((num_nodes, 1), float(step == seq_len - 1), device=device)
                
                # hop_dist
                if hasattr(wkr_dataset, 'hop_matrix'):
                    if not hasattr(wkr_dataset, '_cached_max_h'):
                        _hm = wkr_dataset.hop_matrix
                        wkr_dataset._cached_max_h = float(_hm[_hm < float('inf')].max().item()) if _hm.numel() > 0 else 50.0
                        wkr_dataset._cached_hm_dev = _hm.to(device)
                    hop_dists = wkr_dataset._cached_hm_dev[tgt_node_idx]
                    hop_dist_feat = torch.clamp(hop_dists.float() / wkr_dataset._cached_max_h, 0.0, 1.0).unsqueeze(1)
                else:
                    hop_dist_feat = torch.zeros(num_nodes, 1, device=device)
                    
                # [v3] x,y 제거, time_pct 추가
                time_pct_feat = torch.full((num_nodes, 1), float(step) / max(seq_len, 1), device=device)
                worker_in = torch.cat([flags, net_dist, direction, is_final, hop_dist_feat, time_pct_feat], dim=1)
                
                # edge_dim=3: [length, capacity, speed] (정규화 완료)
                scores, h, c, _ = worker.predict_next_hop(worker_in, edge_index, h, c, batch_vec, edge_attr=_eval_ea)
                pred = scores.argmax().item()
                
                if pred == target_label:
                    wkr_correct += 1
                wkr_total += 1
                
                # Autoregressive update for next step
                curr_node_idx = pred
            wkr_total += 1
        
        # [Check] Save Checkpoint every 20 epochs (총 에폭이 20 이하면 final만 저장)
        if args.epochs > 20 and epoch % 20 == 0:
            ckpt_path = os.path.join(save_dir, f"model_sl_epoch{epoch}.pt")
            checkpoint_data = {
                'manager_state': manager.state_dict(),
                'worker_state': worker.state_dict(),
                'max_dist': max_dist,
                'num_nodes': num_nodes,
                'epoch': epoch
            }
            torch.save(checkpoint_data, ckpt_path)
            print(f"💾 Saved checkpoint with metadata to {ckpt_path}")
    
    tqdm.write(f"📈 Manager: Exact={mgr_exact}/{mgr_total} ({100*mgr_exact/max(1,mgr_total):.1f}%), Near(1-hop)={mgr_near}/{mgr_total} ({100*mgr_near/max(1,mgr_total):.1f}%)")
    tqdm.write(f"📈 Worker: Correct={wkr_correct}/{wkr_total} ({100*wkr_correct/max(1,wkr_total):.1f}%)")
    
    
    # === Success Rate (Simulation-based) ===
    print("\n🎮 Simulating Episodes for Success Rate...", flush=True)
    
    from src.envs.disaster_env import DisasterEnv
    import random
    
    # [Fix] Remove try-except to expose runtime errors
    # [Fix] Correct DisasterEnv instantiation
    env = DisasterEnv(f'data/{args.map}_node.tntp', f'data/{args.map}_net.tntp', device='cpu', verbose=False)
    
    success_count = 0
    total_episodes = 50
    max_steps = 300  # Anaheim(416 nodes) 대응: 100 → 300
    
    # 노드 매핑
    nodes = list(env.map_core.graph.nodes())
    checkpoints = list(range(len(nodes)))  # All node indices as valid checkpoints
    
    # [Optimization] Pre-compute APSP normalization if not already done
    if 'max_dist' not in locals():
        max_dist = apsp_matrix.max().item()
        
    # [Fix #1] 시뮬레이션에서도 이미 GPU에 있는 apsp_device_global을 재사용 (3중 복사 차단)
    apsp_device = apsp_device_global
    pyg_x_device = pyg_data.x.to(device)
    node_coords_device = pyg_x_device[:, :2] # [N, 2]
    edge_index_device = pyg_data.edge_index.to(device)
    edge_attr_device = pyg_data.edge_attr[:, [0, 7, 8]].to(device)  # [E, 3] length+capacity+speed
    # [v3] Min-Max 정규화
    _ea_min = edge_attr_device.min(dim=0, keepdim=True)[0]
    _ea_max = edge_attr_device.max(dim=0, keepdim=True)[0]
    edge_attr_device = (edge_attr_device - _ea_min) / (_ea_max - _ea_min).clamp(min=1e-8)
    batch_vec_sim = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    # [Fix] Wrap Simulation in no_grad to prevent VRAM leak
    with torch.no_grad():
        for ep in range(total_episodes):
            # [Fix] Use env.reset() to generate valid Start/Goal pairs
            env.reset()
            start_idx_val = env.current_node[0].item()
            goal_idx_val = env.target_node[0].item()
            
            start_node = env.idx_to_node[start_idx_val]
            goal_node = env.idx_to_node[goal_idx_val]
            
            # Manager 계획 생성
            # Manager Input: [x, y, is_start, is_goal]
            full_x = torch.zeros((num_nodes, 4), device=device)
            full_x[:, :2] = node_coords_device # Copy coords
            
            start_idx = env.node_mapping[start_node]
            goal_idx = env.node_mapping[goal_node]
            
            full_x[start_idx, 2] = 1 # is_start
            full_x[goal_idx, 3] = 1  # is_goal
            
            # Manager.generate()
            # Masking 파라미터 전달 (apsp_matrix, node_positions)
            sequences, _ = manager.generate(full_x, edge_index_device, batch_vec_sim, valid_tokens=checkpoints, max_len=20, apsp_matrix=hop_apsp_device, node_positions=node_coords_device, edge_attr=edge_attr_device)
            # [Fix] EOS is N (num_nodes), PAD_TOKEN check is redundant for indices but kept for safety
            plan = [t.item() if hasattr(t, 'item') else t for t in sequences[0] if t < num_nodes and t != manager.PAD_TOKEN]
            
            if not plan:
                plan = [goal_idx]
            
            # Worker 실행
            current_node = start_node
            subgoal_idx = 0
            hid = torch.zeros(1, args.hidden_dim, device=device)
            cell = torch.zeros(1, args.hidden_dim, device=device)
            
            # [v3] x,y 제거: 좌표 할당 없음. time_pct는 스텝별 갱신
            worker_in_sim = torch.zeros((num_nodes, 8), device=device)
            
            for step in range(max_steps):
                if current_node == goal_node:
                    success_count += 1
                    break
                
                # 현재 subgoal
                if subgoal_idx < len(plan):
                    subgoal = plan[subgoal_idx]
                    if env.idx_to_node[subgoal] == current_node:
                        subgoal_idx += 1
                        if subgoal_idx < len(plan):
                            subgoal = plan[subgoal_idx]
                        else:
                            subgoal = goal_idx
                else:
                    subgoal = goal_idx
                
                # [Fix C] In-place 업데이트 (새 텐서 할당 없음)
                curr_idx = env.node_mapping[current_node]
                
                # [v3] 인덱스 시프트: x,y 제거로 is_current(0), is_target(1), is_final(5)
                worker_in_sim[:, 0:2].zero_()
                worker_in_sim[:, 5] = 0.0
                worker_in_sim[curr_idx, 0] = 1.0
                if subgoal < num_nodes:
                    worker_in_sim[subgoal, 1] = 1.0
                if subgoal == goal_idx:
                    worker_in_sim[:, 5] = 1.0
                
                # Network Distance (Normalized) → col 2
                worker_in_sim[:, 2] = apsp_device[:, subgoal] / max_dist
                
                # Direction Vector → col 3:5
                target_pos = node_coords_device[subgoal]
                diff = target_pos - node_coords_device
                dist_euc = torch.norm(diff, dim=1, keepdim=True) + 1e-6
                worker_in_sim[:, 3:5] = diff / dist_euc
                
                # hop_dist → col 6
                if hasattr(env, 'hop_matrix'):
                    if not hasattr(env, '_cached_max_h'):
                        _hm = env.hop_matrix
                        env._cached_max_h = float(_hm[_hm < float('inf')].max().item()) if _hm.numel() > 0 else 50.0
                        env._cached_hm_dev = _hm.to(device)
                    hop_dists_sim = env._cached_hm_dev[subgoal]
                    worker_in_sim[:, 6] = torch.clamp(hop_dists_sim.float() / env._cached_max_h, 0.0, 1.0)
                
                # [v3] time_pct → col 7
                worker_in_sim[:, 7] = float(step) / float(max_steps)
                
                # Worker 예측 (predict_next_hop API 사용)
                scores, hid, cell, _ = worker.predict_next_hop(worker_in_sim, edge_index_device, hid, cell, batch_vec_sim, edge_attr=edge_attr_device)
                
                # 이웃 노드 중에서만 선택
                neighbors = list(env.map_core.graph.neighbors(current_node))
                if not neighbors:
                    break
                    
                neighbor_indices = [env.node_mapping[n] for n in neighbors]
                neighbor_scores = scores[neighbor_indices]
                best_neighbor_idx = neighbor_indices[neighbor_scores.argmax().item()]
                next_node = env.idx_to_node[best_neighbor_idx]
                
                current_node = next_node
    
    success_rate = 100 * success_count / total_episodes
    tqdm.write(f"\n🎯 [Manager+Worker] Success Rate: {success_count}/{total_episodes} ({success_rate:.1f}%)")
    
    # === Manager-Only 시뮬레이션 (A* 기반) ===
    # Manager 계획의 품질만 독립 평가: 서브골 간 이동은 A* 최단경로 사용
    tqdm.write("\n🔬 Manager-Only Evaluation (A* Navigation)...")
    
    mgr_only_success = 0
    mgr_plan_quality_sum = 0.0  # 경로 길이 비율 합산
    mgr_plan_quality_count = 0
    
    with torch.no_grad():
        for ep in range(total_episodes):
            env.reset()
            start_node = env.idx_to_node[env.current_node[0].item()]
            goal_node = env.idx_to_node[env.target_node[0].item()]
            
            start_idx = env.node_mapping[start_node]
            goal_idx = env.node_mapping[goal_node]
            
            # Manager 계획 생성 (동일 로직)
            full_x = torch.zeros((num_nodes, 4), device=device)
            full_x[:, :2] = node_coords_device
            full_x[start_idx, 2] = 1
            full_x[goal_idx, 3] = 1
            
            sequences, _ = manager.generate(full_x, edge_index_device, batch_vec_sim, 
                                            valid_tokens=checkpoints, max_len=20, 
                                            apsp_matrix=hop_apsp_device, node_positions=node_coords_device,
                                            edge_attr=edge_attr_device)
            plan = [t.item() if hasattr(t, 'item') else t for t in sequences[0] if t < num_nodes and t != manager.PAD_TOKEN]
            
            if not plan:
                plan = [goal_idx]
            
            # A* 기반 서브골 순차 이동
            current_node = start_node
            reached_goal = False
            total_path_cost = 0.0
            
            waypoints = [start_idx] + plan + [goal_idx]
            
            for i in range(len(waypoints) - 1):
                from_node = env.idx_to_node[waypoints[i]]
                to_node = env.idx_to_node[waypoints[i + 1]]
                
                if from_node == to_node:
                    continue
                
                # A* 최단 경로로 이동 가능한지 확인
                try:
                    path = nx.shortest_path(env.map_core.graph, from_node, to_node, weight='weight')
                    path_cost = nx.shortest_path_length(env.map_core.graph, from_node, to_node, weight='weight')
                    total_path_cost += path_cost
                    current_node = to_node
                except nx.NetworkXNoPath:
                    break  # 경로 없으면 실패
            
            if current_node == goal_node:
                mgr_only_success += 1
                
                # 경로 품질: Manager 경유 경로 / 직접 A* 최적 경로
                try:
                    optimal_cost = nx.shortest_path_length(env.map_core.graph, start_node, goal_node, weight='weight')
                    if optimal_cost > 0:
                        mgr_plan_quality_sum += total_path_cost / optimal_cost
                        mgr_plan_quality_count += 1
                except nx.NetworkXNoPath:
                    pass
    
    mgr_only_rate = 100 * mgr_only_success / total_episodes
    avg_path_ratio = mgr_plan_quality_sum / max(1, mgr_plan_quality_count)
    tqdm.write(f"🔬 [Manager-Only] Success Rate: {mgr_only_success}/{total_episodes} ({mgr_only_rate:.1f}%)")
    tqdm.write(f"📏 [Manager-Only] Avg Path Ratio (vs A*): {avg_path_ratio:.2f}x")
    
    if mgr_only_rate >= 70:
        tqdm.write("✅ Manager 계획 품질 양호. Worker가 병목이라면 RL Fine-tuning 진행.")
    else:
        tqdm.write("⚠️ Manager 계획 품질 부족. Manager 학습 추가 필요.")
    
    # === 학습 곡선 그래프 생성 (논문용: 타이틀 제거, 폰트 확대, 개별 저장) ===
    import matplotlib
    matplotlib.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5,
    })
    
    epochs_range = range(1, len(history['mgr_train_loss']) + 1)
    
    # --- Manager 단독 차트 ---
    fig_mgr, ax_mgr = plt.subplots(figsize=(8, 6))
    ax_mgr.plot(epochs_range, history['mgr_train_loss'], 'b-', label='Train Loss')
    ax_mgr.plot(epochs_range, history['mgr_val_loss'], 'r--', label='Val Loss')
    ax_mgr.set_xlabel('Epoch')
    ax_mgr.set_ylabel('Loss')
    ax_mgr.legend()
    ax_mgr.grid(True, alpha=0.3)
    fig_mgr.tight_layout()
    mgr_path = os.path.join(save_dir, 'sl_curve_manager.png')
    fig_mgr.savefig(mgr_path, dpi=300, bbox_inches='tight')
    plt.close(fig_mgr)
    tqdm.write(f"📈 Manager learning curve saved to {mgr_path}")
    
    # --- Worker 단독 차트 ---
    fig_wkr, ax_wkr = plt.subplots(figsize=(8, 6))
    wkr_train_epochs = list(range(WORKER_FREQ, args.epochs + 1, WORKER_FREQ))
    wkr_train_epochs = wkr_train_epochs[:len(history['wkr_train_loss'])]
    if history['wkr_train_loss']:
        ax_wkr.plot(wkr_train_epochs, history['wkr_train_loss'], 'b-o', label='Train Loss', markersize=6)
        ax_wkr.plot(wkr_train_epochs, history['wkr_val_loss'], 'r--o', label='Val Loss', markersize=6)
    ax_wkr.set_xlabel('Epoch')
    ax_wkr.set_ylabel('Loss')
    ax_wkr.legend()
    ax_wkr.grid(True, alpha=0.3)
    fig_wkr.tight_layout()
    wkr_path = os.path.join(save_dir, 'sl_curve_worker.png')
    fig_wkr.savefig(wkr_path, dpi=300, bbox_inches='tight')
    plt.close(fig_wkr)
    tqdm.write(f"📈 Worker learning curve saved to {wkr_path}")
    
    # --- 합본 차트 (기존 호환) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(epochs_range, history['mgr_train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs_range, history['mgr_val_loss'], 'r--', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if history['wkr_train_loss']:
        axes[1].plot(wkr_train_epochs, history['wkr_train_loss'], 'b-o', label='Train Loss', markersize=6)
        axes[1].plot(wkr_train_epochs, history['wkr_val_loss'], 'r--o', label='Val Loss', markersize=6)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'sl_learning_curve.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"📈 Combined learning curve saved to {curve_path}")
    
    # [Memory Optimization] Aggressive Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # [Fix] Save essential metadata into checkpoint for the RL evaluation
    checkpoint_data = {
        'manager_state': manager.state_dict(),
        'worker_state': worker.state_dict(),
        'max_dist': max_dist,
        'num_nodes': num_nodes,
        'epoch': args.epochs
    }
    torch.save(checkpoint_data, os.path.join(save_dir, "model_sl_final.pt"))
    print(f"Saved models and metadata to {save_dir}/model_sl_final.pt")

    # === 학습 결과 Summary 자동 생성 ===
    _generate_training_summary(
        save_dir=save_dir,
        args=args,
        history=history,
        best_mgr_val=best_mgr_val,
        best_wkr_val=best_wkr_val,
        wkr_frozen=wkr_frozen,
        mgr_lr_final=optimizer_mgr.param_groups[0]['lr'],
        wkr_lr_final=wkr_opt.param_groups[0]['lr'],
        num_nodes=num_nodes,
        max_dist=max_dist,
    )


def _generate_training_summary(
    save_dir: str,
    args,
    history: dict,
    best_mgr_val: float,
    best_wkr_val: float,
    wkr_frozen: bool,
    mgr_lr_final: float,
    wkr_lr_final: float,
    num_nodes: int,
    max_dist: float,
) -> None:
    """SL 학습 완료 후 주요 지표를 텍스트 파일로 저장하는 유틸리티 함수."""
    from datetime import datetime as dt

    # Manager 지표 계산
    mgr_train = history.get('mgr_train_loss', [])
    mgr_val = history.get('mgr_val_loss', [])
    mgr_train_first = mgr_train[0] if mgr_train else float('nan')
    mgr_train_last = mgr_train[-1] if mgr_train else float('nan')
    mgr_val_first = mgr_val[0] if mgr_val else float('nan')
    mgr_val_last = mgr_val[-1] if mgr_val else float('nan')
    mgr_best_epoch = (mgr_val.index(min(mgr_val)) + 1) if mgr_val else 0

    # Worker 지표 계산
    wkr_train = history.get('wkr_train_loss', [])
    wkr_val = history.get('wkr_val_loss', [])
    wkr_train_first = wkr_train[0] if wkr_train else float('nan')
    wkr_train_last = wkr_train[-1] if wkr_train else float('nan')
    wkr_val_first = wkr_val[0] if wkr_val else float('nan')
    wkr_val_last = wkr_val[-1] if wkr_val else float('nan')
    wkr_epochs_trained = len(wkr_train)

    # 과적합 판정: Train < Val 이고 차이가 일정 이상
    mgr_overfit = (mgr_train_last < mgr_val_last * 0.7) if mgr_train else False
    wkr_overfit = (wkr_train_last < wkr_val_last * 0.7) if wkr_train else False

    lines = [
        "=" * 72,
        " SL Pre-training Summary Report",
        f" Generated: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "■ 실험 설정",
        f"  Map:        {args.map} ({num_nodes} nodes)",
        f"  Epochs:     {args.epochs}",
        f"  Batch Size: {args.batch_size}",
        f"  Hidden Dim: {args.hidden_dim}",
        f"  Edge Dim:   1 (Phase 1: length only)",
        f"  Max Dist:   {max_dist:.2f}",
        f"  Manager LR: {args.lr_manager} → {mgr_lr_final:.6f} (final)",
        f"  Worker LR:  {args.lr_worker} → {wkr_lr_final:.6f} (final)",
        "",
        "■ Manager 학습 결과",
        f"  Train Loss:    {mgr_train_first:.4f} → {mgr_train_last:.4f}",
        f"  Val Loss:      {mgr_val_first:.4f} → {mgr_val_last:.4f}",
        f"  Best Val Loss: {best_mgr_val:.4f} (Epoch {mgr_best_epoch})",
        f"  과적합 여부:   {'⚠️ 의심' if mgr_overfit else '✅ 정상'}",
        "",
        "■ Worker 학습 결과",
        f"  Train Loss:    {wkr_train_first:.4f} → {wkr_train_last:.4f}" if wkr_train else "  Train Loss:    N/A",
        f"  Val Loss:      {wkr_val_first:.4f} → {wkr_val_last:.4f}" if wkr_val else "  Val Loss:      N/A",
        f"  Best Val Loss: {best_wkr_val:.4f}",
        f"  학습 에포크:   {wkr_epochs_trained}",
        f"  Early Stop:    {'⛔ 발동 (Frozen)' if wkr_frozen else '미발동'}",
        f"  과적합 여부:   {'⚠️ 의심' if wkr_overfit else '✅ 정상'}",
        "",
        "■ 핵심 지표 요약",
        f"  Manager Best Val:    {best_mgr_val:.4f}",
        f"  Worker Best Val:     {best_wkr_val:.4f}",
        f"  Manager Train/Val:   {mgr_train_last:.4f} / {mgr_val_last:.4f}  (gap: {abs(mgr_train_last - mgr_val_last):.4f})",
        f"  Worker Frozen:       {wkr_frozen}",
        f"  Manager LR 감소:     {'감소함' if mgr_lr_final < args.lr_manager * 0.99 else '미감소 (지속 개선)'}",
        "",
        "=" * 72,
    ]

    summary_path = os.path.join(save_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"📋 Training summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Anaheim', help='Map Name (e.g. Anaheim, SiouxFalls)')
    parser.add_argument('--data', type=str, default='data', help='Data Directory containing .pt files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (DAgger: 20 recommended)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (Reduced to 32 for VRAM safety)')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr_manager', type=float, default=5e-4, help='Manager Learning Rate (default: 5e-4)')
    parser.add_argument('--lr_worker', type=float, default=5e-5, help='Worker Learning Rate (default: 5e-5)')
    parser.add_argument('--worker_only', action='store_true', help='Train Worker only')
    parser.add_argument('--parallel', action='store_true', help='Use parallel multi-threading for SL training (Default: False)')

    args = parser.parse_args()
    train_sl(args)
