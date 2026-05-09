"""
Cross-Map Zero-shot 평가 스크립트.

Anaheim에서 학습된 Worker 모델을 다른 맵에서 재학습 없이 평가.
Zone 파일이 필요하며, generate_zones.py로 사전 생성해야 합니다.

사용법:
  python scripts/evaluate_crossmap.py --map SiouxFalls --checkpoint logs/.../best.pt --num_episodes 200
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.envs.hrl_env import HRLZoneEnv
from src.models.worker import Worker


def evaluate(
    env: HRLZoneEnv,
    worker: Worker,
    device: torch.device,
    num_episodes: int = 200,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Worker 모델의 Zero-shot 성능 평가.
    
    Args:
        env: 평가 대상 환경
        worker: 학습된 Worker 모델
        device: 디바이스
        num_episodes: 총 평가 에피소드 수
        batch_size: 배치 크기
    Returns:
        metrics: {'success_rate', 'avg_reward', 'avg_path_len', ...}
    """
    worker.eval()  # 평가 모드
    
    # 정적 edge_index 생성
    edge_list = []
    for u, v in env.G.edges():
        edge_list.append((env.node_to_idx[u], env.node_to_idx[v]))
    edge_list_bidir = edge_list + [(v, u) for u, v in edge_list]
    edge_index = torch.tensor(edge_list_bidir, dtype=torch.long).t().to(device)
    
    # 정적 edge_attr 미리 계산 (use_edge_attr=True 인 경우)
    static_edge_attr = None
    if getattr(worker, 'use_edge_attr', False) or getattr(worker, 'config', {}).get('use_edge_attr', False) or True:
        curr_edge_attr = []
        for u_idx, v_idx in edge_list_bidir:
            u = env.idx_to_node[u_idx]
            v = env.idx_to_node[v_idx]
            data = env.dm.graph[u][v]
            length = data.get('length', 0.0)
            capacity = data.get('capacity', 0.0)
            speed = data.get('speed', 0.0)
            curr_edge_attr.append([length, capacity, speed])
        static_edge_attr = torch.tensor(curr_edge_attr, dtype=torch.float).to(device)
        
        # Min-Max 정규화 (학습 시와 동일)
        if static_edge_attr.size(0) > 0:
            feat_min = static_edge_attr.min(dim=0, keepdim=True)[0]
            feat_max = static_edge_attr.max(dim=0, keepdim=True)[0]
            scale = (feat_max - feat_min).clamp(min=1e-8)
            static_edge_attr = (static_edge_attr - feat_min) / scale
    
    N = env.num_nodes
    all_results: List[Dict] = []
    
    num_batches = (num_episodes + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            actual_batch = min(batch_size, num_episodes - batch_idx * batch_size)
            state = env.reset(batch_size=actual_batch)
            B = actual_batch
            
            # 배치별 결과 추적
            batch_rewards = [0.0] * B
            done_flags = [False] * B
            final_infos = [{} for _ in range(B)]
            
            while not all(done_flags):
                active = [b for b in range(B) if not done_flags[b]]
                
                active_states = torch.stack(
                    [state[b].to(device) for b in active]
                )
                active_masks = torch.stack(
                    [env.get_action_mask_batch()[b].to(device) for b in active]
                )
                
                A = len(active)
                x_flat = active_states.view(-1, active_states.shape[-1])
                mask_flat = active_masks.view(-1)
                ai = torch.arange(A, device=device).repeat_interleave(N)
                aei = torch.cat([edge_index + i * N for i in range(A)], dim=1)
                
                # Edge 속성 추출 (사전 계산된 값 반복)
                edge_attr_flat = None
                if static_edge_attr is not None:
                    edge_attr_flat = static_edge_attr.repeat(A, 1)
                
                probs_all, _, _ = worker(x_flat, aei, edge_attr=edge_attr_flat, batch=ai, neighbors_mask=mask_flat)
                
                # Greedy 선택 (평가이므로 argmax)
                actions = []
                for i, b in enumerate(active):
                    node_probs = probs_all[i * N: (i + 1) * N]
                    action = node_probs.argmax().item()
                    actions.append(action)
                
                all_actions = []
                ai_ptr = 0
                for b in range(B):
                    if not done_flags[b]:
                        all_actions.append(actions[ai_ptr])
                        ai_ptr += 1
                    else:
                        all_actions.append(0)
                
                next_state, reward_t, done_t, infos = env.step_batch(
                    torch.tensor(all_actions)
                )
                
                for b in range(B):
                    if not done_flags[b]:
                        batch_rewards[b] += reward_t[b].item()
                        if done_t[b].item():
                            done_flags[b] = True
                            final_infos[b] = infos[b]
                
                state = next_state
            
            # 배치 결과 수집
            for b in range(B):
                all_results.append({
                    'success': 1.0 if final_infos[b].get('reason') == 'success' else 0.0,
                    'reward': batch_rewards[b],
                    'path_len': final_infos[b].get('path_len', 200),
                    'reason': final_infos[b].get('reason', 'unknown'),
                })
    
    # 통계 계산
    success_rate = np.mean([r['success'] for r in all_results])
    avg_reward = np.mean([r['reward'] for r in all_results])
    avg_path_len = np.mean([r['path_len'] for r in all_results])
    
    # 성공한 에피소드만의 평균 경로 길이
    success_lens = [r['path_len'] for r in all_results if r['success'] > 0]
    avg_success_path = np.mean(success_lens) if success_lens else 0.0
    
    # 실패 원인 분포
    reason_counts = defaultdict(int)
    for r in all_results:
        reason_counts[r['reason']] += 1
    
    metrics = {
        'success_rate': float(success_rate),
        'avg_reward': float(avg_reward),
        'avg_path_len': float(avg_path_len),
        'avg_success_path': float(avg_success_path),
        'num_episodes': len(all_results),
        'reason_distribution': dict(reason_counts),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Cross-Map Zero-shot 평가')
    parser.add_argument('--map', type=str, required=True, help='평가 맵')
    parser.add_argument('--checkpoint', type=str, required=True, help='Worker 체크포인트 경로')
    parser.add_argument('--num_episodes', type=int, default=200, help='평가 에피소드 수')
    parser.add_argument('--num_layers', type=int, default=3, help='GATv2 레이어 수')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden 차원')
    parser.add_argument('--batch_size', type=int, default=16, help='평가 배치 크기')
    # 환경 설정 (학습과 동일하게)
    parser.add_argument('--masking_mode', type=str, default='soft_curr_next')
    parser.add_argument('--use_pbrs', action='store_true')
    parser.add_argument('--subgoal_mode', type=str, default='zone', choices=['zone', 'node'])
    parser.add_argument('--use_jk_net', action='store_true')
    parser.add_argument('--use_edge_attr', action='store_true')
    # 무시되는 인자 (호환성)
    parser.add_argument('--use_gae', action='store_true')
    parser.add_argument('--entropy_coeff', type=float, default=0.0)
    parser.add_argument('--zone_progress_reward', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Zone 파일 자동 탐색
    zone_json = 'data/node_to_zone_k30.json'  # 기본값
    zone_graph_json = 'data/zone_graph_k30.json'
    map_zone_files = glob.glob(f"data/node_to_zone_{args.map}_k*.json")
    if map_zone_files:
        zone_json = sorted(map_zone_files)[-1]
        k_val = zone_json.split('_k')[-1].replace('.json', '')
        zone_graph_json = f"data/zone_graph_{args.map}_k{k_val}.json"
    
    print(f"🗺️  맵: {args.map}")
    print(f"   Zone: {zone_json}")
    print(f"   체크포인트: {args.checkpoint}")
    print(f"   에피소드: {args.num_episodes}")
    
    # 환경 생성
    env = HRLZoneEnv(
        f"data/{args.map}_node.tntp",
        f"data/{args.map}_net.tntp",
        zone_json=zone_json,
        zone_graph_json=zone_graph_json,
        masking_mode=args.masking_mode,
        use_pbrs=args.use_pbrs,
        subgoal_mode=args.subgoal_mode,
    )
    print(f"   노드: {env.num_nodes}, Zone: {env.k}")
    
    # Worker 모델 로드
    worker = Worker(
        node_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_jk_net=args.use_jk_net,
        use_edge_attr=args.use_edge_attr,
        dropout=0.0,  # 평가 시 dropout 비활성
        use_checkpoint=False,
    ).to(device)
    
    # 체크포인트 로드
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'worker_state_dict' in payload:
        state_dict = payload['worker_state_dict']
    elif 'worker_state' in payload:
        state_dict = payload['worker_state']
    elif 'model_state_dict' in payload:
        state_dict = payload['model_state_dict']
    else:
        state_dict = payload
    
    # 호환성: strict=False로 키 불일치 허용
    missing, unexpected = worker.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   ⚠️ Missing keys: {len(missing)}")
    if unexpected:
        print(f"   ⚠️ Unexpected keys: {len(unexpected)}")
    
    print(f"\n📊 평가 시작...")
    metrics = evaluate(env, worker, device, args.num_episodes, args.batch_size)
    
    # 결과 출력
    print(f"\n{'='*50}")
    print(f"📊 {args.map} Zero-shot 평가 결과")
    print(f"{'='*50}")
    print(f"  Success Rate:    {metrics['success_rate']*100:.1f}%")
    print(f"  Avg Reward:      {metrics['avg_reward']:.1f}")
    print(f"  Avg Path Len:    {metrics['avg_path_len']:.1f}")
    print(f"  Avg Success Path:{metrics['avg_success_path']:.1f}")
    print(f"  Episodes:        {metrics['num_episodes']}")
    print(f"  Failure Reasons: {metrics['reason_distribution']}")
    
    # JSON 결과 저장
    result_path = f"logs/eval_zeroshot_{args.map}_{args.subgoal_mode}_results.json"
    with open(result_path, 'w') as f:
        json.dump({'map': args.map, **metrics}, f, indent=2)
    print(f"\n  ✅ 결과 저장: {result_path}")


if __name__ == '__main__':
    main()
