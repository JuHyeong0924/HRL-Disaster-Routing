"""
eval_worker_sanity.py — Worker 아키텍처 검증 통합 진단 스크립트
실험 1: Oracle Worker (APSP 정답) 상한선 검증
실험 2: Worker Error Analysis (오답 패턴 분석)
"""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.disaster_env import DisasterEnv
from src.models.worker import WorkerLSTM


def run_oracle_experiment(env: DisasterEnv, episodes: int = 500) -> float:
    """실험 1: Oracle Worker (APSP Next-Hop) 이론적 상한선 검증.
    
    Worker 자리에 APSP 정답 테이블을 직접 꽂아서
    Manager 계획 없이 순수 네비게이션 상한선을 측정한다.
    """
    print(f"\n{'='*60}")
    print(f"[실험 1] Oracle Worker 상한선 검증 ({episodes} episodes)")
    print(f"{'='*60}")
    
    successes = 0
    total_steps_list = []
    
    for _ in tqdm(range(episodes), desc="Oracle Sim"):
        env.reset(batch_size=1, sync_problem=False)
        goal = env.target_node.item()
        start = env.current_node.item()
        
        if start == goal:
            successes += 1
            total_steps_list.append(0)
            continue
        
        steps = 0
        for steps in range(300):  # max 300 steps
            curr = env.current_node.item()
            if curr == goal:
                successes += 1
                break
            
            # Oracle: hop 기반 최단경로 다음 노드 직접 선택
            next_hop = env.hop_next_hop_matrix[curr, goal].item()
            if next_hop < 0:
                break  # 경로 없음
            
            env.step(torch.tensor([next_hop], device=env.device))
        
        total_steps_list.append(steps + 1)
    
    oracle_rate = successes / episodes * 100
    avg_steps = np.mean(total_steps_list) if total_steps_list else 0
    
    print(f"\n🎯 Oracle Worker 성공률: {successes}/{episodes} ({oracle_rate:.1f}%)")
    print(f"📏 평균 스텝 수: {avg_steps:.1f}")
    print(f"   → 기대치: 95%+ (이하면 Manager 계획 품질 재검토 필요)")
    
    return oracle_rate


def run_error_analysis(
    env: DisasterEnv,
    worker: WorkerLSTM,
    episodes: int = 500,
    hidden_dim: int = 256,
) -> dict:
    """실험 2: Worker Error Analysis.
    
    Worker가 틀릴 때의 패턴을 분석:
    - 시퀀스 위치별 (초반 vs 후반)
    - 서브골 홉 거리별 (근거리 vs 원거리)
    - 오답 노드의 정답 대비 홉 거리
    - 이웃 노드 중 정답 선택률
    """
    print(f"\n{'='*60}")
    print(f"[실험 2] Worker Error Analysis ({episodes} episodes)")
    print(f"{'='*60}")
    
    worker.eval()
    
    # 통계 수집기
    stats = defaultdict(int)
    error_hop_distances = []  # 오답과 정답 사이의 홉 거리
    position_bins = {"0-25%": [0, 0], "25-50%": [0, 0], "50-75%": [0, 0], "75-100%": [0, 0]}
    hop_bins = {"1-2hop": [0, 0], "3-4hop": [0, 0], "5-8hop": [0, 0], "9+hop": [0, 0]}
    neighbor_correct = 0  # 이웃 노드 내에서의 정답 수
    neighbor_total = 0    # 이웃 노드 내에서의 총 판정 수
    
    with torch.no_grad():
        for ep_i in tqdm(range(episodes), desc="Error Analysis"):
            env.reset(batch_size=1, sync_problem=False)
            goal = env.target_node.item()
            start = env.current_node.item()
            
            if start == goal:
                continue
            
            h = torch.zeros(1, hidden_dim, device=env.device)
            c_state = torch.zeros(1, hidden_dim, device=env.device)
            total_hops = max(int(env.hop_matrix[start, goal].item()), 1)
            
            for step in range(min(total_hops * 3, 100)):  # 최적 경로의 3배까지만
                curr = env.current_node.item()
                if curr == goal:
                    break
                
                # Oracle 정답
                expert_next = env.hop_next_hop_matrix[curr, goal].item()
                if expert_next < 0:
                    break
                
                # Worker 입력 조립 (v4.1 호환: env.pyg_data.x에서 7-Dim 추출)
                x = env.pyg_data.x
                if x.size(1) < 10:
                    pad_size = 10 - x.size(1)
                    x = torch.cat([x, torch.zeros(x.size(0), pad_size, device=env.device)], dim=1)
                
                # [is_curr(2), is_tgt(3), dist(5), dir_x(6), dir_y(7), is_final(8), hop(9)]
                spatial = torch.cat([x[:, 2:4], x[:, 5:]], dim=1)  # [N, 7]
                
                time_pct = min(step / 100.0, 1.0)
                time_to_go = torch.tensor([[1.0 - time_pct]], device=env.device)
                
                # 엣지 피처 Min-Max 정규화
                edge_attr = env.pyg_data.edge_attr
                if edge_attr is not None and edge_attr.size(1) >= 9:
                    selected = edge_attr[:, [0, 7, 8]]
                    f_min = selected.min(dim=0, keepdim=True)[0]
                    scale = (selected.max(dim=0, keepdim=True)[0] - f_min).clamp(min=1e-8)
                    edge_attr_norm = (selected - f_min) / scale
                else:
                    edge_attr_norm = None
                
                scores, h, c_state, _ = worker.predict_next_hop(
                    x=spatial,
                    edge_index=env.pyg_data.edge_index,
                    h_state=h, c_state=c_state,
                    batch=env.pyg_data.batch,
                    time_to_go=time_to_go,
                    detach_spatial=True,
                    edge_attr=edge_attr_norm,
                )
                
                # 이웃 노드 마스킹 후 선택
                mask = env.get_mask().bool()
                masked = scores.view(1, -1).masked_fill(~mask, -1e9)
                pred = masked.argmax(dim=-1).item()
                
                stats["total_steps"] += 1
                
                # 이웃 노드 내 정답 분석
                valid_neighbors = mask[0].nonzero(as_tuple=True)[0].tolist()
                if expert_next in valid_neighbors:
                    neighbor_total += 1
                    if pred == expert_next:
                        neighbor_correct += 1
                
                # 시퀀스 위치 분류
                progress = step / max(total_hops, 1)
                if progress < 0.25:
                    pos_key = "0-25%"
                elif progress < 0.5:
                    pos_key = "25-50%"
                elif progress < 0.75:
                    pos_key = "50-75%"
                else:
                    pos_key = "75-100%"
                position_bins[pos_key][1] += 1  # 총 수
                
                # 서브골 홉 거리 분류
                hop_remains = int(env.hop_matrix[curr, goal].item())
                if hop_remains <= 2:
                    hop_key = "1-2hop"
                elif hop_remains <= 4:
                    hop_key = "3-4hop"
                elif hop_remains <= 8:
                    hop_key = "5-8hop"
                else:
                    hop_key = "9+hop"
                hop_bins[hop_key][1] += 1  # 총 수
                
                if pred != expert_next:
                    stats["total_err"] += 1
                    position_bins[pos_key][0] += 1  # 에러 수
                    hop_bins[hop_key][0] += 1
                    
                    # 오답 노드와 정답 노드 사이의 홉 거리
                    err_dist = int(env.hop_matrix[pred, expert_next].item())
                    error_hop_distances.append(err_dist)
                
                # 모델 예측대로 진행 (autoregressive)
                env.step(torch.tensor([pred], device=env.device))
    
    # === 결과 출력 ===
    total_steps = max(stats["total_steps"], 1)
    total_err = max(stats["total_err"], 1)
    err_rate = stats["total_err"] / total_steps * 100
    
    print(f"\n📊 전체 오류율: {stats['total_err']}/{total_steps} ({err_rate:.1f}%)")
    print(f"   정확도: {100-err_rate:.1f}%")
    
    print(f"\n📈 시퀀스 위치별 정확도 (LSTM 장기 기억력 검증):")
    for key, (errs, total) in position_bins.items():
        acc = (1 - errs / max(total, 1)) * 100
        print(f"   {key:>8s}: {acc:5.1f}% ({total-errs}/{total})")
    
    print(f"\n📈 서브골 홉 거리별 정확도 (GATv2 공간 인코딩 범위 검증):")
    for key, (errs, total) in hop_bins.items():
        acc = (1 - errs / max(total, 1)) * 100
        print(f"   {key:>7s}: {acc:5.1f}% ({total-errs}/{total})")
    
    if error_hop_distances:
        arr = np.array(error_hop_distances)
        print(f"\n📈 오답 노드의 정답 대비 홉 거리 분포:")
        print(f"   평균: {arr.mean():.2f} hop, 중앙값: {np.median(arr):.1f} hop")
        print(f"   1-hop 이내 오차: {(arr <= 1).sum()}/{len(arr)} ({(arr <= 1).mean()*100:.1f}%)")
        print(f"   2-hop 이내 오차: {(arr <= 2).sum()}/{len(arr)} ({(arr <= 2).mean()*100:.1f}%)")
        print(f"   5-hop 이상 오차: {(arr >= 5).sum()}/{len(arr)} ({(arr >= 5).mean()*100:.1f}%)")
    
    if neighbor_total > 0:
        nbr_acc = neighbor_correct / neighbor_total * 100
        print(f"\n📈 이웃 노드 중 정답 선택률 (SL/평가 분포 불일치 검증):")
        print(f"   {neighbor_correct}/{neighbor_total} ({nbr_acc:.1f}%)")
    
    return dict(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker 아키텍처 검증 통합 진단")
    parser.add_argument("--map", type=str, default="Anaheim")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--checkpoint", type=str,
                        default="logs/sl_pretrain/2026-04-23_1342_sl_ep50/model_sl_final.pt")
    args = parser.parse_args()
    
    print(f"🖥️  Device: {args.device}")
    print(f"🗺️  Map: {args.map}")
    print(f"📦 Checkpoint: {args.checkpoint}")
    
    env = DisasterEnv(
        f"data/{args.map}_node.tntp",
        f"data/{args.map}_net.tntp",
        device=args.device,
        verbose=True,
    )
    
    # === 실험 1: Oracle Worker ===
    oracle_rate = run_oracle_experiment(env, episodes=args.episodes)
    
    # === 실험 2: Worker Error Analysis ===
    if os.path.exists(args.checkpoint):
        worker = WorkerLSTM(
            node_dim=7,
            hidden_dim=args.hidden_dim,
            edge_dim=3,
        ).to(args.device)
        
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        worker.load_state_dict(ckpt["worker_state"])
        print(f"✅ Worker 가중치 로드 완료 (epoch={ckpt.get('epoch', '?')})")
        
        run_error_analysis(env, worker, episodes=args.episodes, hidden_dim=args.hidden_dim)
    else:
        print(f"⚠️ 체크포인트 없음: {args.checkpoint}")
        print("   Worker Error Analysis 건너뜀.")
    
    # === 최종 판정 ===
    print(f"\n{'='*60}")
    print(f"📋 최종 판정")
    print(f"{'='*60}")
    if oracle_rate >= 95:
        print(f"✅ Oracle {oracle_rate:.1f}% → Worker 개선 여지 명확. RL Fine-tuning으로 갭 해소 가능.")
    elif oracle_rate >= 80:
        print(f"⚠️ Oracle {oracle_rate:.1f}% → Worker 개선 여지 있으나 Manager 계획도 일부 검토 필요.")
    else:
        print(f"❌ Oracle {oracle_rate:.1f}% → Manager 계획 자체에 문제. Worker보다 Manager 재학습 우선.")
