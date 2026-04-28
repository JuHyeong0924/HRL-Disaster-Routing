#!/usr/bin/env python3
"""
Zone Clustering for Anaheim Map (METIS-based Balanced Graph Partitioning)
=========================================================================
METIS 알고리즘으로 균등 크기의 Zone 분할 수행.
- 구역당 목표 노드 수(zone_size) 기반으로 K 자동 결정
- Edge-cut 최소화 → 구역 내부 연결성 극대화
- Zone-Level Coarsened Graph 생성 (Manager 입력용)

사용법:
    python zone_clustering.py --zone_size 14
    python zone_clustering.py --k 30
"""
import os
import sys
import json
import math
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Dict, List, Tuple, Optional
import pymetis

# 프로젝트 루트 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.envs.disaster_map import DisasterMap


# ──────────────────────────────────────────────
#  1. 맵 로드 & APSP 계산
# ──────────────────────────────────────────────
def load_anaheim_graph() -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """Anaheim 맵 로드"""
    node_file = os.path.join('data', 'Anaheim_node.tntp')
    net_file = os.path.join('data', 'Anaheim_net.tntp')
    dm = DisasterMap(node_file, net_file)
    return dm.graph, dm.pos


def compute_apsp_hop_matrix(
    G: nx.Graph,
    node_to_idx: Dict[int, int]
) -> np.ndarray:
    """APSP 홉 거리 행렬 계산"""
    n = len(G.nodes())
    hop_matrix = np.full((n, n), np.inf)
    apsp = dict(nx.all_pairs_shortest_path_length(G))
    for u, lengths in apsp.items():
        u_idx = node_to_idx[u]
        for v, hop_val in lengths.items():
            v_idx = node_to_idx[v]
            hop_matrix[u_idx, v_idx] = hop_val
    return hop_matrix


# ──────────────────────────────────────────────
#  2. METIS 기반 균등 분할
# ──────────────────────────────────────────────
def metis_partition(
    G: nx.Graph,
    node_to_idx: Dict[int, int],
    k: int
) -> np.ndarray:
    """
    METIS 그래프 파티셔닝 수행.
    
    METIS는 edge-cut을 최소화하면서 균등 크기 파티션을 생성하는
    다단계(Multi-Level) 그래프 분할 알고리즘.
    
    Args:
        G: NetworkX 그래프
        node_to_idx: 노드 ID → 인덱스 매핑
        k: 파티션(구역) 수
        
    Returns:
        labels: [N] 각 노드의 구역 라벨 (0~K-1)
    """
    n = len(G.nodes())
    nodes = sorted(G.nodes())
    
    # METIS 입력 형식: adjacency list (인덱스 기반)
    # pymetis.part_graph(nparts, adjacency=[list_of_neighbors_for_each_node])
    adjacency = [[] for _ in range(n)]
    for u, v in G.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        adjacency[u_idx].append(v_idx)
        adjacency[v_idx].append(u_idx)
    
    # METIS 파티셔닝 수행
    # edge-cut 최소화 + 균등 크기 보장
    edge_cuts, membership = pymetis.part_graph(k, adjacency=adjacency)
    
    labels = np.array(membership)
    print(f"   METIS edge-cuts: {edge_cuts} (구역 간 잘린 에지 수)")
    
    return labels


# ──────────────────────────────────────────────
#  3. Zone-Level Coarsened Graph 생성 (Manager 입력)
# ──────────────────────────────────────────────
def build_zone_graph(
    G: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    labels: np.ndarray,
    node_to_idx: Dict[int, int],
    idx_to_node: Dict[int, int],
    hop_matrix: np.ndarray,
    k: int
) -> Dict:
    """
    Zone-Level Coarsened Graph 생성.
    
    Manager에게 제공할 축약 그래프:
    - Zone Node: 구역별 집계 피처 (중심 좌표, 노드 수, 내부 연결도, 평균 hop)
    - Zone Edge: 구역 간 연결 에지 수, 평균 capacity
    - Zone Adjacency: 어떤 구역이 어떤 구역과 인접한지
    
    Returns:
        zone_graph_data: dict 형태의 Zone Graph 정보
    """
    # ── Zone Node Features ──
    zone_nodes = {}
    for zone_id in range(k):
        zone_indices = np.where(labels == zone_id)[0]
        zone_node_ids = [idx_to_node[i] for i in zone_indices]
        
        # 구역 중심 좌표
        zone_positions = np.array([pos[nid] for nid in zone_node_ids])
        center_x = float(np.mean(zone_positions[:, 0]))
        center_y = float(np.mean(zone_positions[:, 1]))
        
        # 구역 내부 연결도 (내부 에지 수 / 최대 가능 에지 수)
        subgraph = G.subgraph(zone_node_ids)
        n_internal_edges = subgraph.number_of_edges()
        n_nodes = len(zone_node_ids)
        max_edges = n_nodes * (n_nodes - 1) / 2
        internal_density = n_internal_edges / max_edges if max_edges > 0 else 0.0
        
        # 구역 내부 평균/최대 홉 거리
        if n_nodes > 1:
            sub_hops = hop_matrix[np.ix_(zone_indices, zone_indices)]
            valid = sub_hops[sub_hops > 0]
            avg_diameter = float(np.mean(valid)) if len(valid) > 0 else 0.0
            max_diameter = float(np.max(valid)) if len(valid) > 0 else 0.0
        else:
            avg_diameter = 0.0
            max_diameter = 0.0
        
        # 구역 내 평균 capacity/speed
        caps, spds = [], []
        for u, v, d in subgraph.edges(data=True):
            caps.append(d.get('capacity', 0.0))
            spds.append(d.get('speed', 0.0))
        avg_capacity = float(np.mean(caps)) if caps else 0.0
        avg_speed = float(np.mean(spds)) if spds else 0.0
        
        zone_nodes[zone_id] = {
            'node_ids': [int(nid) for nid in zone_node_ids],
            'n_nodes': n_nodes,
            'center': [center_x, center_y],
            'internal_density': round(internal_density, 3),
            'avg_diameter_hop': round(avg_diameter, 2),
            'max_diameter_hop': round(max_diameter, 2),
            'avg_capacity': round(avg_capacity, 1),
            'avg_speed': round(avg_speed, 1),
            'n_internal_edges': n_internal_edges
        }
    
    # ── Zone Edge Features (구역 간 연결 정보) ──
    zone_edges = {}
    zone_adjacency = {z: set() for z in range(k)}
    
    for u, v, d in G.edges(data=True):
        u_zone = int(labels[node_to_idx[u]])
        v_zone = int(labels[node_to_idx[v]])
        
        # 서로 다른 구역 간 에지만 (inter-zone)
        if u_zone != v_zone:
            edge_key = (min(u_zone, v_zone), max(u_zone, v_zone))
            if edge_key not in zone_edges:
                zone_edges[edge_key] = {
                    'n_bridges': 0,  # 구역 간 연결 도로 수
                    'capacities': [],
                    'speeds': [],
                    'is_highways': [],
                    'bridge_nodes': []  # 경계 노드 쌍
                }
            
            zone_edges[edge_key]['n_bridges'] += 1
            zone_edges[edge_key]['capacities'].append(d.get('capacity', 0.0))
            zone_edges[edge_key]['speeds'].append(d.get('speed', 0.0))
            zone_edges[edge_key]['is_highways'].append(d.get('is_highway', 0.0))
            zone_edges[edge_key]['bridge_nodes'].append([int(u), int(v)])
            
            zone_adjacency[u_zone].add(v_zone)
            zone_adjacency[v_zone].add(u_zone)
    
    # 집계
    zone_edges_summary = {}
    for (z1, z2), info in zone_edges.items():
        zone_edges_summary[f"{z1}-{z2}"] = {
            'zones': [z1, z2],
            'n_bridges': info['n_bridges'],
            'avg_capacity': round(float(np.mean(info['capacities'])), 1),
            'avg_speed': round(float(np.mean(info['speeds'])), 1),
            'has_highway': int(any(h > 0 for h in info['is_highways'])),
            'bridge_nodes': info['bridge_nodes']
        }
    
    # Zone 간 APSP (Zone 중심 노드 기반)
    zone_hop_matrix = np.zeros((k, k))
    for z1 in range(k):
        z1_indices = np.where(labels == z1)[0]
        for z2 in range(k):
            if z1 == z2:
                continue
            z2_indices = np.where(labels == z2)[0]
            # 두 구역 간 최소 홉 거리
            cross_hops = hop_matrix[np.ix_(z1_indices, z2_indices)]
            zone_hop_matrix[z1, z2] = float(np.min(cross_hops))
    
    return {
        'k': k,
        'zone_nodes': zone_nodes,
        'zone_edges': zone_edges_summary,
        'zone_adjacency': {int(z): sorted(list(adj)) for z, adj in zone_adjacency.items()},
        'zone_hop_matrix': zone_hop_matrix.tolist()
    }


# ──────────────────────────────────────────────
#  4. 시각화 (논문 Figure 품질)
# ──────────────────────────────────────────────
def visualize_zones(
    G: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    labels: np.ndarray,
    node_to_idx: Dict[int, int],
    idx_to_node: Dict[int, int],
    zone_data: Dict,
    k: int,
    output_path: str
):
    """METIS 구역화 결과 시각화"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # 균등 분포 색상 (K개)
    colors = plt.cm.hsv(np.linspace(0, 0.95, k))
    
    # 1. 에지 그리기
    edge_segments = []
    edge_colors_list = []
    for u, v in G.edges():
        u_pos = np.array(pos[u])
        v_pos = np.array(pos[v])
        edge_segments.append([u_pos, v_pos])
        
        u_zone = labels[node_to_idx[u]]
        v_zone = labels[node_to_idx[v]]
        if u_zone == v_zone:
            c = colors[u_zone]
            edge_colors_list.append((*c[:3], 0.4))
        else:
            # 구역 경계 에지 = 빨간 점선 스타일
            edge_colors_list.append((1.0, 1.0, 1.0, 0.12))
    
    lc = LineCollection(edge_segments, colors=edge_colors_list, linewidths=0.8)
    ax.add_collection(lc)
    
    # 2. 노드 그리기
    for node in G.nodes():
        idx = node_to_idx[node]
        zone_id = labels[idx]
        x, y = pos[node]
        degree = G.degree(node)
        size = max(15, min(60, degree * 8))
        ax.scatter(x, y, c=[colors[zone_id]], s=size,
                  edgecolors='white', linewidths=0.3, zorder=3, alpha=0.85)
    
    # 3. 구역 라벨
    zone_nodes = zone_data['zone_nodes']
    for zone_id in range(k):
        info = zone_nodes[zone_id]
        cx, cy = info['center']
        n = info['n_nodes']
        ax.text(cx, cy, f'Z{zone_id}\n({n})',
               fontsize=6, fontweight='bold', color='white',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[zone_id],
                        alpha=0.7, edgecolor='white', linewidth=0.5),
               zorder=5)
    
    # 4. 타이틀
    sizes = [zone_nodes[z]['n_nodes'] for z in range(k)]
    stats_text = (
        f"Nodes: {len(G.nodes())}  |  Edges: {len(G.edges())}  |  "
        f"Zones: {k}  |  Nodes/Zone: {np.mean(sizes):.1f}±{np.std(sizes):.1f}  |  "
        f"Range: [{min(sizes)}, {max(sizes)}]"
    )
    
    ax.set_title(
        f'Anaheim Network — METIS Balanced Partitioning (K={k})',
        fontsize=14, fontweight='bold', color='white', pad=15
    )
    ax.text(0.5, -0.02, stats_text, transform=ax.transAxes,
           fontsize=10, color='#aaaaaa', ha='center')
    ax.set_xlabel('X Coordinate', fontsize=10, color='#888888')
    ax.set_ylabel('Y Coordinate', fontsize=10, color='#888888')
    ax.tick_params(colors='#666666', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"🖼️ 시각화 저장: {output_path}")


# ──────────────────────────────────────────────
#  5. Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='METIS 기반 Zone Clustering')
    parser.add_argument('--zone_size', type=int, default=14,
                       help='구역당 목표 노드 수 (default: 14)')
    parser.add_argument('--k', type=int, default=None,
                       help='구역 수 직접 지정 (zone_size보다 우선)')
    parser.add_argument('--output_dir', type=str, default='data')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 맵 로드
    print("🌍 Anaheim 맵 로드 중...")
    G, pos = load_anaheim_graph()
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    print(f"   ✅ 노드: {n_nodes}개, 에지: {n_edges}개")
    
    # 노드 매핑
    nodes_sorted = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_sorted)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 2. K 결정 (zone_size 기반 또는 직접 지정)
    if args.k is not None:
        k = args.k
        zone_size = math.ceil(n_nodes / k)
        print(f"\n📐 K={k} 직접 지정 → 구역당 ~{zone_size}개 노드")
    else:
        zone_size = args.zone_size
        k = math.ceil(n_nodes / zone_size)
        print(f"\n📐 zone_size={zone_size} → K={k} 자동 결정 ({n_nodes}/{zone_size})")
    
    # 3. APSP 홉 거리
    print("\n🔗 APSP 홉 거리 행렬 계산 중...")
    hop_matrix = compute_apsp_hop_matrix(G, node_to_idx)
    avg_hop = np.mean(hop_matrix[hop_matrix < np.inf])
    print(f"   ✅ 평균 홉 거리: {avg_hop:.1f}")
    
    # 4. METIS 파티셔닝
    print(f"\n⚡ METIS 파티셔닝 (K={k})...")
    labels = metis_partition(G, node_to_idx, k)
    
    # 통계
    sizes = [int(np.sum(labels == z)) for z in range(k)]
    print(f"   ✅ 구역당 노드: min={min(sizes)}, max={max(sizes)}, "
          f"avg={np.mean(sizes):.1f}, std={np.std(sizes):.1f}")
    
    # 5. Zone Graph 생성 (Manager 입력)
    print("\n🏗️ Zone-Level Coarsened Graph 생성 중...")
    zone_data = build_zone_graph(G, pos, labels, node_to_idx, idx_to_node, hop_matrix, k)
    
    n_zone_edges = len(zone_data['zone_edges'])
    avg_bridges = np.mean([e['n_bridges'] for e in zone_data['zone_edges'].values()])
    print(f"   ✅ Zone 에지: {n_zone_edges}개, 구역 간 평균 연결 도로: {avg_bridges:.1f}개")
    
    # 상세 출력
    print(f"\n📊 구역별 상세:")
    print(f"{'Zone':>6} | {'Nodes':>5} | {'Density':>7} | {'AvgDiam':>7} | {'MaxDiam':>7} | {'IntEdges':>8} | {'Neighbors':>9}")
    print('-' * 75)
    for z in range(k):
        info = zone_data['zone_nodes'][z]
        n_adj = len(zone_data['zone_adjacency'][z])
        print(f"  Z{z:>3} | {info['n_nodes']:>5} | {info['internal_density']:>7.3f} | "
              f"{info['avg_diameter_hop']:>7.1f} | {info['max_diameter_hop']:>7.0f} | "
              f"{info['n_internal_edges']:>8} | {n_adj:>9}")
    
    # 6. 저장
    # node_to_zone 매핑
    node_to_zone = {}
    for idx, zone_id in enumerate(labels):
        node_id = idx_to_node[idx]
        node_to_zone[int(node_id)] = int(zone_id)
    
    json_path = os.path.join(output_dir, f'node_to_zone_k{k}.json')
    with open(json_path, 'w') as f:
        json.dump(node_to_zone, f, indent=2)
    print(f"\n💾 node_to_zone 저장: {json_path}")
    
    # Zone Graph (Manager 입력 데이터)
    zone_graph_path = os.path.join(output_dir, f'zone_graph_k{k}.json')
    with open(zone_graph_path, 'w') as f:
        json.dump(zone_data, f, indent=2, default=str)
    print(f"💾 Zone Graph 저장: {zone_graph_path}")
    
    # 7. 시각화
    fig_path = os.path.join(output_dir, f'zone_metis_k{k}.png')
    visualize_zones(G, pos, labels, node_to_idx, idx_to_node, zone_data, k, fig_path)
    
    print(f"\n🎉 완료!")
    print(f"   - 구역 매핑: {json_path}")
    print(f"   - Zone Graph: {zone_graph_path}")
    print(f"   - 시각화: {fig_path}")


if __name__ == '__main__':
    main()
