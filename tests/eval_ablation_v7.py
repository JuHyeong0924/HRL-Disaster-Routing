"""
Worker Ablation V7.1 통합 평가 스크립트
지표: SR, AvgPL (hop), AvgSP_hop, Ratio_hop, AvgTotalDist, AvgSP_dist, Ratio_dist
"""
import os, sys, json, torch, numpy as np, networkx as nx, random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.worker_env import WorkerEnv
from src.models.worker import Worker


def _load_ckpt(path, worker, device):
    if not os.path.exists(path):
        return False
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict):
        state = payload.get('worker_state', payload.get('state_dict', payload))
    else:
        state = payload
    current = worker.state_dict()
    compat = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    worker.load_state_dict(compat, strict=False)
    return True


EXPERIMENTS = {
    # V7 원본 (hop 실험 — 코드 변경 없음)
    'C_hop_tern_uni': {
        'ckpt': 'logs/rl_worker_stage/2026-06-15_000323_worker_B32_ablV7_C_hop_ternary_uniform/best.pt',
        'dist_mode': 'hop', 'zone_info_mode': 'ternary', 'zone_weight_mode': 'uniform',
        'desc': 'hop + ternary + uniform',
    },
    'D_hop_bin_euc': {
        'ckpt': 'logs/rl_worker_stage/2026-06-15_000922_worker_B32_ablV7_D_hop_binary_euclidean/best.pt',
        'dist_mode': 'hop', 'zone_info_mode': 'binary', 'zone_weight_mode': 'euclidean',
        'desc': 'hop + binary + euclidean',
    },
    # V7.1 (log1p dijkstra + F)
    'Bv2_dij_bin_uni': {
        'ckpt': 'logs/rl_worker_stage/2026-06-15_003616_worker_B32_ablV71_Bv2_dijkstra_binary_uniform/best.pt',
        'dist_mode': 'dijkstra', 'zone_info_mode': 'binary', 'zone_weight_mode': 'uniform',
        'desc': 'dijkstra(log1p) + binary + uniform',
    },
    'Ev2_dij_tern_euc': {
        'ckpt': 'logs/rl_worker_stage/2026-06-15_003616_worker_B32_ablV71_Ev2_dijkstra_ternary_euclidean/best.pt',
        'dist_mode': 'dijkstra', 'zone_info_mode': 'ternary', 'zone_weight_mode': 'euclidean',
        'desc': 'dijkstra(log1p) + ternary + euclidean',
    },
    'F_hop_tern_euc': {
        'ckpt': 'logs/rl_worker_stage/2026-06-15_004238_worker_B32_ablV71_F_hop_ternary_euclidean/best.pt',
        'dist_mode': 'hop', 'zone_info_mode': 'ternary', 'zone_weight_mode': 'euclidean',
        'desc': 'hop + ternary + euclidean (최종 후보)',
    },
}

NUM_EPISODES = 200
BATCH_SIZE = 16
MAP = 'Anaheim'
SEED = 42


def evaluate(exp_name, cfg, device):
    print(f"\n{'─'*60}")
    print(f"📊 {exp_name}: {cfg['desc']}")

    if not os.path.exists(cfg['ckpt']):
        print(f"   ❌ 체크포인트 없음"); return None

    env = WorkerEnv(
        f"data/{MAP}_node.tntp", f"data/{MAP}_net.tntp",
        zone_json=f"data/grid_{MAP}_node_to_zone.json",
        zone_graph_json=f"data/grid_{MAP}_zone_graph.json",
        masking_mode='soft_curr_next', use_pbrs=True, use_is_visited=True,
        use_relative_hop=True, oob_penalty=-1.0,
        dist_mode=cfg['dist_mode'], zone_info_mode=cfg['zone_info_mode'],
        zone_weight_mode=cfg['zone_weight_mode'],
    )

    worker = Worker(node_dim=5, hidden_dim=256, num_layers=2, dropout=0.0, use_is_visited=True).to(device)
    _load_ckpt(cfg['ckpt'], worker, device)
    worker.eval()

    # 그래프 구조
    edge_list = [(env.node_to_idx[u], env.node_to_idx[v]) for u, v in env.G.edges()]
    bidir = edge_list + [(v, u) for u, v in edge_list]
    edge_index = torch.tensor(bidir, dtype=torch.long).t().to(device)
    N = env.num_nodes

    # Dijkstra 최단 거리 사전 계산 (모든 쌍)
    sp_dist_dict = dict(nx.all_pairs_dijkstra_path_length(env.G, weight='weight'))

    np.random.seed(SEED); random.seed(SEED)
    results = []

    with torch.no_grad():
        for bi in range((NUM_EPISODES + BATCH_SIZE - 1) // BATCH_SIZE):
            B = min(BATCH_SIZE, NUM_EPISODES - bi * BATCH_SIZE)
            state = env.reset(batch_size=B)

            # Shortest path (hop + dist)
            sp_hops, sp_dists = [], []
            for b in range(B):
                s = env.idx_to_node[int(env.curr_nodes[b])]
                t = env.idx_to_node[int(env.target_nodes[b])]
                try:
                    sp_hops.append(len(nx.shortest_path(env.G, s, t)) - 1)
                except:
                    sp_hops.append(999)
                try:
                    sp_dists.append(float(sp_dist_dict[s][t]))
                except:
                    sp_dists.append(float('inf'))

            done_flags = [False] * B
            final_infos = [{}] * B

            for step in range(200):
                if all(done_flags): break
                active = [b for b in range(B) if not done_flags[b]]
                A = len(active)
                xs = torch.stack([state[b].to(device) for b in active])
                ms = torch.stack([env.get_action_mask_batch()[b].to(device) for b in active])
                xf = xs.view(-1, xs.shape[-1])
                mf = ms.view(-1)
                ai = torch.arange(A, device=device).repeat_interleave(N)
                aei = torch.cat([edge_index + i * N for i in range(A)], dim=1)
                probs, _, _ = worker(xf, aei, edge_attr=None, batch=ai, neighbors_mask=mf)
                acts = [probs[i*N:(i+1)*N].argmax().item() for i in range(A)]
                all_acts = []; ptr = 0
                for b in range(B):
                    if not done_flags[b]: all_acts.append(acts[ptr]); ptr += 1
                    else: all_acts.append(0)
                state, _, d, infos = env.step_batch(torch.tensor(all_acts))
                for b in range(B):
                    if not done_flags[b] and d[b].item():
                        done_flags[b] = True; final_infos[b] = infos[b]

            for b in range(B):
                s = 1.0 if final_infos[b].get('reason') == 'success' else 0.0
                pl = final_infos[b].get('path_len', 200)
                td = final_infos[b].get('total_dist', 0.0)
                results.append({
                    'success': s, 'path_len': pl, 'total_dist': td,
                    'sp_hop': sp_hops[b], 'sp_dist': sp_dists[b],
                    'ratio_hop': pl / max(sp_hops[b], 1) if s > 0 else float('inf'),
                    'ratio_dist': td / sp_dists[b] if s > 0 and sp_dists[b] > 0 else float('inf'),
                    'reason': final_infos[b].get('reason', 'unknown'),
                })

    succ = [r for r in results if r['success'] > 0]
    sr = np.mean([r['success'] for r in results]) * 100
    avg_pl = np.mean([r['path_len'] for r in succ]) if succ else 0
    avg_td = np.mean([r['total_dist'] for r in succ]) if succ else 0
    avg_sp_h = np.mean([r['sp_hop'] for r in succ]) if succ else 0
    avg_sp_d = np.mean([r['sp_dist'] for r in succ]) if succ else 0
    ratio_h = np.mean([r['ratio_hop'] for r in succ]) if succ else 0
    ratio_d = np.mean([r['ratio_dist'] for r in succ]) if succ else 0
    reasons = Counter(r['reason'] for r in results)

    m = {
        'exp_name': exp_name, 'desc': cfg['desc'],
        'success_rate': round(sr, 1),
        'avg_path_len': round(avg_pl, 1), 'avg_sp_hop': round(avg_sp_h, 1), 'ratio_hop': round(ratio_h, 3),
        'avg_total_dist': round(avg_td, 1), 'avg_sp_dist': round(avg_sp_d, 1), 'ratio_dist': round(ratio_d, 3),
        'reasons': dict(reasons),
    }
    print(f"   SR={sr:.1f}% | PL={avg_pl:.1f}(SP_h={avg_sp_h:.1f}, R_h={ratio_h:.3f}) | "
          f"Dist={avg_td:.0f}(SP_d={avg_sp_d:.0f}, R_d={ratio_d:.3f})")
    return m


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔬 V7.1 통합 평가 (Map: {MAP}, {NUM_EPISODES}ep)")

    all_m = {}
    for name in sorted(EXPERIMENTS):
        r = evaluate(name, EXPERIMENTS[name], device)
        if r: all_m[name] = r

    print(f"\n{'='*100}")
    print(f"📊 V7.1 결과 요약")
    print(f"{'='*100}")
    print(f"{'ID':<22} {'dist':>8} {'zone':>7} {'zw':>9} {'SR':>5} {'PL':>5} {'SP_h':>5} {'R_hop':>6} {'TDist':>7} {'SP_d':>7} {'R_dist':>7}")
    print(f"{'─'*100}")
    for n, m in sorted(all_m.items()):
        c = EXPERIMENTS[n]
        print(f"{n:<22} {c['dist_mode']:>8} {c['zone_info_mode']:>7} {c['zone_weight_mode']:>9} "
              f"{m['success_rate']:>4.0f}% {m['avg_path_len']:>5.1f} {m['avg_sp_hop']:>5.1f} {m['ratio_hop']:>6.3f} "
              f"{m['avg_total_dist']:>7.0f} {m['avg_sp_dist']:>7.0f} {m['ratio_dist']:>7.3f}")

    out = 'tests/ablation_results/v7_dist_zone/ablation_v71_results.json'
    with open(out, 'w') as f:
        json.dump(all_m, f, indent=2, ensure_ascii=False)
    print(f"\n💾 {out}")


if __name__ == '__main__':
    main()
