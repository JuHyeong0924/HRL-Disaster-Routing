import os
import sys
from datetime import datetime

# [Warning Suppression] Filter PyTorch Prototype Warnings
import warnings
warnings.filterwarnings("ignore", ".*nested_tensor.*", category=UserWarning)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Active Device: {device}")
    
    # MEMORY OPTIMIZATION: Try loading Pre-processed Tensors
    # MEMORY OPTIMIZATION: Enforce Pre-processed Tensors
    # We REMOVE the fallback to pickle to prevent accidental 23GB memory usage.
    
    pt_manager_path = 'data/manager_data.pt'
    pt_worker_path = 'data/worker_data.pt'

    # 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                
    if os.path.exists(pt_manager_path) and os.path.exists(pt_worker_path):
        print("✅ Loading Optimized Tensor Data (.pt)...", flush=True)
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
    print("🌍 Computing APSP for Worker features...", flush=True)
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
    print(f"✅ APSP Ready. Max Distance: {max_dist:.2f}", flush=True)

    print("🔗 Computing Hop APSP for Manager masking...", flush=True)
    hop_apsp = dict(nx.all_pairs_shortest_path_length(G))
    hop_apsp_matrix = torch.full((num_nodes, num_nodes), float('inf'), dtype=torch.float32)
    for u_id, lengths in hop_apsp.items():
        u_idx = node_to_idx[u_id]
        for v_id, hop_val in lengths.items():
            v_idx = node_to_idx[v_id]
            hop_apsp_matrix[u_idx, v_idx] = float(hop_val)
    hop_apsp_device = hop_apsp_matrix.to(device)
    print("✅ Hop APSP Ready.", flush=True)
    
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
    print(f"✅ Next Hop Table Ready. Shape: {apsp_next_hop.shape}", flush=True)
    
    # [Fix #1] GPU 캐싱 먼저 수행 (학습 루프에서 실제로 사용할 텐서)
    apsp_device_global = apsp_matrix.to(device)
    
    mgr_dataset = HierarchicalDataset(mgr_samples, pyg_data, mode='manager')
    
    # Worker에 APSP 행렬 전달 (net_dist, dir_x, dir_y 피처 계산용)
    wkr_dataset = HierarchicalDataset(wkr_samples, pyg_data, mode='worker',
                                       apsp_matrix=apsp_matrix, max_dist=max_dist)
    
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
    
    print(f"📊 Manager: Train={mgr_train_size}, Val={mgr_val_size}")
    print(f"📊 Worker: Train={wkr_train_size}, Val={wkr_val_size}")
    
    # Clean up
    if 'data_dict' in locals():
        del data_dict
    import gc; gc.collect()
    
    # DATA LOADER OMPTIMIZATION:
    num_workers = 0 # Force single process for debug/stability on Windows 
    print(f"Using {num_workers} DataLoader workers (Optimal for Mem/Speed)...")
    
    # Train Loaders
    mgr_loader = DataLoader(mgr_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=hierarchical_collate, num_workers=0, pin_memory=True)
    wkr_loader = DataLoader(wkr_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=hierarchical_collate, num_workers=0, pin_memory=True)
    
    # Validation Loaders
    mgr_val_loader = DataLoader(mgr_val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=hierarchical_collate, num_workers=0, pin_memory=True)
    wkr_val_loader = DataLoader(wkr_val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=hierarchical_collate, num_workers=0, pin_memory=True)
    
    manager = GraphTransformerManager(
        node_dim=4, 
        hidden_dim=args.hidden_dim, 
        dropout=0.2
    ).to(device)
    
    worker = WorkerLSTM(
        node_dim=8,  # [x, y, is_current, is_target, net_dist, dir_x, dir_y, is_final_target_phase]
        hidden_dim=args.hidden_dim
    ).to(device)
    
    
    # [Hyperparam] Separate LRs
    optimizer_mgr = optim.Adam(manager.parameters(), lr=args.lr_manager)
    wkr_opt = optim.Adam(worker.parameters(), lr=args.lr_worker)
    
    # Learning Rate Schedulers
    # Manager: ReduceLROnPlateau (Plateau detection)
    mgr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mgr, mode='min', factor=0.5, patience=5)
    
    # Worker: Cosine Annealing (Stable decay)
    wkr_scheduler = optim.lr_scheduler.CosineAnnealingLR(wkr_opt, T_max=args.epochs, eta_min=args.lr_worker * 0.01)
    
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
    
    # [Fix #1] APSP 행렬은 이미 GPU에 캐싱 완료 (apsp_device_global)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        
        # --- Manager Train ---
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
            node_emb_all = manager.topology_enc(batch.x, batch.edge_index) # [B*N, Hidden]
            
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
            pointer_logits = manager(batch.x, batch.edge_index, batch.batch, target_seq_emb=target_seq_emb, edge_attr=batch.edge_attr)
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
                log_probs = F.log_softmax(pointer_logits, dim=-1) # [B, L+1, N+1]
                
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
                
                # Exclude EOS token from logic (it has no coordinates)
                # Compare only node distributions (N_max classes)
                log_probs_nodes = log_probs[..., :-1] # [B, L+1, N_max]
                
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
            
        print(f"Manager Loss: {mgr_loss/max(1, mgr_batches):.4f}", flush=True)
            
        # --- Worker Train ---
        # [Improved] Worker 학습 빈도 2배 증가 (20에폭에서 10회)
        WORKER_FREQ = 2
        
        # [DAgger] TF Ratio 감소: 0.3까지 감소 (원래 경로 감독 유지)
        # 0.0으로 내리면 CE/KL 충돌로 성능 하락 → 70% 자율 + 30% Teacher로 균형
        tf_ratio = max(0.3, 1.0 - 1.5 * (epoch - 1) / args.epochs)
        window_size = 5 # 다중 스텝 언롤링(Multi-step Unrolling) 윈도우 크기
        
        if WORKER_FREQ > 0 and epoch % WORKER_FREQ == 0:
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
                t_seqs_pad = pad_sequence(t_seqs, batch_first=True, padding_value=0)
                n_seqs_pad = pad_sequence(n_seqs, batch_first=True, padding_value=0)
                seq_lens = torch.tensor([seq.size(0) for seq in c_seqs], dtype=torch.long, device=device)
                
                max_seq_len = c_seqs_pad.size(1)
                
                loss_buffer = [] # 윈도우 손실 버퍼
                
                # 이전 스텝의 예측 노드 (Autoregressive 용도)
                prev_preds = c_seqs_pad[:, 0].clone() # [B]
                
                # [Optimization v3] 동적 배치 축소를 위한 기본 변수 사전 정의 (루프 밖)
                node_coords_per_graph = batch.x[:, :2].view(num_graphs, -1, 2)  # [B, N, 2]
                num_nodes_per_graph = node_coords_per_graph.size(1)
                base_edges = pyg_data.edge_index.to(device)  # [2, E_single]
                base_edge_attr = pyg_data.edge_attr.to(device)  # [E_single, edge_dim]
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
                    # Teacher Forcing이 아닌 경우 → 현재 위치에서 서브골까지의 최단경로 다음 노드
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
                    worker_in[:, :2] = compact_coords
                    worker_in[cached_offsets + curr_nodes_k, 2] = 1.0  # is_current
                    worker_in[cached_offsets + tgt_nodes_k, 3] = 1.0   # is_target
                    
                    # APSP 거리 피처
                    if hasattr(wkr_loader.dataset, 'apsp_matrix'):
                        max_dst = wkr_loader.dataset.max_dist
                        active_dists = apsp_device_global[tgt_nodes_k] / max_dst  # [K, N]
                        worker_in[:, 4] = active_dists.reshape(-1)
                    
                    # 방향 벡터 피처
                    tgt_coords = compact_coords[cached_offsets + tgt_nodes_k]  # [K, 2]
                    tgt_coords_rep = tgt_coords.repeat_interleave(num_nodes_per_graph, dim=0)  # [K*N, 2]
                    diff = tgt_coords_rep - compact_coords
                    norm_val = diff.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    worker_in[:, 5:7] = diff / norm_val
                    is_final_target = (act_step == (seq_lens[active_idx] - 1)).float()
                    worker_in[:, 7] = is_final_target.repeat_interleave(num_nodes_per_graph)
                    
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
            
            print(f"Worker Train Loss: {wkr_loss/max(1, wkr_batches):.4f}, Acc: {wkr_acc:.1f}% (TF Ratio: {tf_ratio:.2f})", flush=True)
        else:
            print(f"Worker Train: Skipped (Freq={WORKER_FREQ})", flush=True)
        
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
                node_emb_all = manager.topology_enc(batch.x, batch.edge_index)
                
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
                pointer_logits = manager(batch.x, batch.edge_index, batch.batch, target_seq_emb=target_seq_emb, edge_attr=batch.edge_attr)
                
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
                    log_probs = F.log_softmax(pointer_logits, dim=-1)
                    
                    real_mask_2d = (final_targets < N_max) & (final_targets >= 0)
                    safe_targets_2d = final_targets.clone()
                    safe_targets_2d[~real_mask_2d] = 0
                    
                    dists = apsp_device_global[safe_targets_2d]
                    
                    temperature = 1.0
                    soft_probs = F.softmax(-dists / temperature, dim=-1)
                    del dists  # [Fix #2] Validation Soft Label 중간 텐서 즉시 해제
                    
                    log_probs_nodes = log_probs[..., :-1]
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
                    base_edge_attr = pyg_data.edge_attr.to(device)
                    
                    # [Fix A] Pre-allocation (Validation) — edge_index/batch_vec만 캐싱
                    cached_K = val_num_graphs
                    cached_offsets = torch.arange(val_num_graphs, device=device) * num_nodes_per_graph
                    cached_edge_offset = cached_offsets.view(val_num_graphs, 1, 1)
                    cached_edge_index = (base_edges.unsqueeze(0) + cached_edge_offset).permute(1, 0, 2).reshape(2, -1)
                    cached_edge_attr = base_edge_attr.repeat(val_num_graphs, 1)
                    cached_batch_vec = torch.arange(val_num_graphs, device=device).repeat_interleave(num_nodes_per_graph)
                    
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
                        worker_in[:, :2] = compact_coords
                        worker_in[cached_offsets + curr_nodes_k, 2] = 1.0
                        worker_in[cached_offsets + tgt_nodes_k, 3] = 1.0
                        
                        # APSP 거리 피처
                        if hasattr(wkr_val_loader.dataset, 'apsp_matrix'):
                            max_dst = wkr_val_loader.dataset.max_dist
                            active_dists = apsp_device_global[tgt_nodes_k] / max_dst
                            worker_in[:, 4] = active_dists.reshape(-1)
                        
                        # 방향 벡터 피처
                        tgt_coords = compact_coords[cached_offsets + tgt_nodes_k]
                        tgt_coords_rep = tgt_coords.repeat_interleave(num_nodes_per_graph, dim=0)
                        diff = tgt_coords_rep - compact_coords
                        norm_val = diff.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        worker_in[:, 5:7] = diff / norm_val
                        is_final_target = (act_step == (seq_lens[active_idx] - 1)).float()
                        worker_in[:, 7] = is_final_target.repeat_interleave(num_nodes_per_graph)
                        
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
        if epoch % WORKER_FREQ == 0:
            wkr_val = wkr_val_loss/max(1, wkr_val_batches)
            print(f"📊 Validation - Manager: {mgr_val:.4f}, Worker: {wkr_val:.4f}", flush=True)
            # wkr_val logging
            history['wkr_train_loss'].append(wkr_loss/max(1, wkr_batches))
            history['wkr_val_loss'].append(wkr_val)
        else:
            print(f"📊 Validation - Manager: {mgr_val:.4f}, Worker: Skipped", flush=True)
            # Worker가 skip된 에폭에는 기록하지 않음 (learning curve에서 학습 에폭만 표시)
            pass
        
        # 학습 곡선 기록
        mgr_train = mgr_loss/max(1, mgr_batches)
        # wkr_train handled above
        history['mgr_train_loss'].append(mgr_train)
        history['mgr_val_loss'].append(mgr_val)
        
        manager.train()
        worker.train()
        
        # Step LR Schedulers
        mgr_scheduler.step(mgr_val) # ReduceLROnPlateau needs metric
        if epoch % WORKER_FREQ == 0:
            wkr_scheduler.step()
        
        # Get LR safely (ReduceLROnPlateau does not have get_last_lr in old versions, use optimizer param_groups)
        current_lr_mgr = optimizer_mgr.param_groups[0]['lr']
        current_lr_wkr = wkr_opt.param_groups[0]['lr']
        print(f"Current LR - Manager: {current_lr_mgr:.6f}, Worker: {current_lr_wkr:.6f}", flush=True)

        # [Fix #5] stale del try/except 제거 (logits, loss는 이 스코프에 존재하지 않음)
        # torch.cuda.empty_cache() — 매 에폭 강제 삭제 방지 (VRAM 캐싱 풀 활용)
    
    # === Accuracy Evaluation ===
    print("\n📊 Evaluating Model Accuracy...", flush=True)
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
            logits = manager(x_in.squeeze(0), edge_index, batch_vec, target_seq_emb=None, edge_attr=pyg_data.edge_attr.to(device))
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
                
                worker_in = torch.cat([node_coords, flags, net_dist, direction], dim=1)
                
                # [Fix] Use pyg_data.edge_attr as edge_attr_device is not yet defined in this scope
                scores, h, c, _ = worker.predict_next_hop(worker_in, edge_index, h, c, batch_vec, edge_attr=pyg_data.edge_attr.to(device))
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
    
    print(f"📈 Manager: Exact={mgr_exact}/{mgr_total} ({100*mgr_exact/max(1,mgr_total):.1f}%), Near(1-hop)={mgr_near}/{mgr_total} ({100*mgr_near/max(1,mgr_total):.1f}%)")
    print(f"📈 Worker: Correct={wkr_correct}/{wkr_total} ({100*wkr_correct/max(1,wkr_total):.1f}%)")
    
    
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
    edge_attr_device = pyg_data.edge_attr.to(device)  # [E, edge_dim] for models
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
            
            # [Fix C] 시뮬레이션 입력 텐서 사전 할당 (torch.cat 제거)
            worker_in_sim = torch.zeros((num_nodes, 8), device=device)
            worker_in_sim[:, :2] = node_coords_device  # 좌표는 변하지 않음
            
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
                
                # is_current, is_target, is_final_target_phase 초기화 및 설정
                worker_in_sim[:, 2:4].zero_()
                worker_in_sim[:, 7] = 0.0
                worker_in_sim[curr_idx, 2] = 1.0
                if subgoal < num_nodes:
                    worker_in_sim[subgoal, 3] = 1.0
                if subgoal == goal_idx:
                    worker_in_sim[:, 7] = 1.0
                
                # Network Distance (Normalized)
                worker_in_sim[:, 4] = apsp_device[:, subgoal] / max_dist
                
                # Direction Vector
                target_pos = node_coords_device[subgoal]
                diff = target_pos - node_coords_device
                dist_euc = torch.norm(diff, dim=1, keepdim=True) + 1e-6
                worker_in_sim[:, 5:7] = diff / dist_euc
                
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
    print(f"\n🎯 [Manager+Worker] Success Rate: {success_count}/{total_episodes} ({success_rate:.1f}%)")
    
    # === Manager-Only 시뮬레이션 (A* 기반) ===
    # Manager 계획의 품질만 독립 평가: 서브골 간 이동은 A* 최단경로 사용
    print("\n🔬 Manager-Only Evaluation (A* Navigation)...", flush=True)
    
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
    print(f"🔬 [Manager-Only] Success Rate: {mgr_only_success}/{total_episodes} ({mgr_only_rate:.1f}%)")
    print(f"📏 [Manager-Only] Avg Path Ratio (vs A*): {avg_path_ratio:.2f}x")
    
    if mgr_only_rate >= 70:
        print("✅ Manager 계획 품질 양호. Worker가 병목이라면 RL Fine-tuning 진행.")
    else:
        print("⚠️ Manager 계획 품질 부족. Manager 학습 추가 필요.")
    
    # === 학습 곡선 그래프 생성 ===
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history['mgr_train_loss']) + 1)
    
    # Manager Loss 곡선
    axes[0].plot(epochs_range, history['mgr_train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, history['mgr_val_loss'], 'r--', label='Val Loss', linewidth=2)
    axes[0].set_title('Manager Learning Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Worker Loss 곡선 — 학습한 에폭만 표시 (skip된 에폭 제외)
    wkr_train_epochs = list(range(WORKER_FREQ, args.epochs + 1, WORKER_FREQ))
    wkr_train_epochs = wkr_train_epochs[:len(history['wkr_train_loss'])]  # 안전 장치
    if history['wkr_train_loss']:
        axes[1].plot(wkr_train_epochs, history['wkr_train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
        axes[1].plot(wkr_train_epochs, history['wkr_val_loss'], 'r--o', label='Val Loss', linewidth=2, markersize=6)
    axes[1].set_title('Worker Learning Curve (CE+KL)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (0.7*CE + 0.3*KL)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'sl_learning_curve.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"📈 Learning curve saved to {curve_path}")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Anaheim', help='Map Name (e.g. Anaheim, SiouxFalls)')
    parser.add_argument('--data', type=str, default='data', help='Data Directory containing .pt files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (DAgger: 20 recommended)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (Reduced to 32 for VRAM safety)')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr_manager', type=float, default=5e-4, help='Manager Learning Rate (default: 5e-4)')
    parser.add_argument('--lr_worker', type=float, default=5e-5, help='Worker Learning Rate (default: 5e-5)')
    
    args = parser.parse_args()
    train_sl(args)
