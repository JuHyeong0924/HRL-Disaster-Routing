"""
맵별 Zone 분할 생성 스크립트.

METIS 그래프 파티셔닝을 사용하여 각 맵에 대해:
  1. node_to_zone_{map_name}_k{K}.json
  2. zone_graph_{map_name}_k{K}.json
을 생성합니다.

사용법:
  python scripts/generate_zones.py                    # 전체 맵
  python scripts/generate_zones.py --map SiouxFalls    # 특정 맵
  python scripts/generate_zones.py --map Anaheim --k 30  # K 지정
"""
import argparse
import json
import os
import sys

import networkx as nx

# pymetis가 없으면 networkx 기반 Kernighan-Lin bisection 사용
try:
    import pymetis
    HAS_PYMETIS = True
except ImportError:
    HAS_PYMETIS = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.envs.disaster_map import DisasterMap


def partition_graph_metis(G: nx.Graph, k: int) -> dict:
    """METIS 기반 K-way 그래프 파티셔닝.
    
    pymetis가 없으면 networkx의 community detection으로 대체.
    
    Args:
        G: NetworkX 그래프
        k: Zone 수
    Returns:
        node_to_zone: {node_id: zone_id} 매핑
    """
    nodes = sorted(list(G.nodes()))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    n = len(nodes)
    
    if HAS_PYMETIS and k > 1:
        # pymetis 형식: adjacency list (각 노드의 이웃 인덱스 리스트)
        adj_list = []
        for i in range(n):
            node = idx_to_node[i]
            neighbors = [node_to_idx[nb] for nb in G.neighbors(node)]
            adj_list.append(neighbors)
        
        # METIS 파티셔닝 수행
        _, membership = pymetis.part_graph(k, adjacency=adj_list)
        node_to_zone = {idx_to_node[i]: int(membership[i]) for i in range(n)}
    else:
        # pymetis 없을 때: Spectral Bisection 기반 재귀 분할
        # 또는 단순 community detection 사용
        from networkx.algorithms.community import greedy_modularity_communities
        
        # greedy_modularity는 k를 직접 지정 못하므로 근사적으로 사용
        # 대안: Louvain community detection
        try:
            from community import best_partition  # python-louvain
            # Louvain은 자동으로 커뮤니티 수를 결정하므로 k와 정확히 맞지 않을 수 있음
            partition = best_partition(G)
            node_to_zone = {n: int(z) for n, z in partition.items()}
        except ImportError:
            # 최후 수단: 균등 분할 (노드 정렬 후 K등분)
            print(f"  ⚠️ pymetis/python-louvain 없음. 균등 분할 사용.")
            chunk_size = max(1, n // k)
            node_to_zone = {}
            for i, node in enumerate(nodes):
                node_to_zone[node] = min(i // chunk_size, k - 1)
    
    return node_to_zone


def build_zone_graph(G: nx.Graph, node_to_zone: dict, k: int) -> dict:
    """Zone 간 인접 관계 그래프 생성.
    
    Args:
        G: 원본 그래프
        node_to_zone: 노드→Zone 매핑
        k: Zone 수
    Returns:
        zone_graph_data: {'k': K, 'zone_adjacency': {zone_id: [neighbors]}}
    """
    # 실제 사용된 Zone ID 집합
    zone_ids = sorted(set(node_to_zone.values()))
    actual_k = len(zone_ids)
    
    # Zone 간 인접 관계 계산
    zone_adj = {z: set() for z in zone_ids}
    for u, v in G.edges():
        zu = node_to_zone.get(u)
        zv = node_to_zone.get(v)
        if zu is not None and zv is not None and zu != zv:
            zone_adj[zu].add(zv)
            zone_adj[zv].add(zu)
    
    # JSON 직렬화를 위해 set → sorted list 변환
    zone_adjacency = {str(z): sorted(list(neighbors)) for z, neighbors in zone_adj.items()}
    
    # Zone별 노드 수 통계
    zone_sizes = {}
    for n, z in node_to_zone.items():
        zone_sizes[z] = zone_sizes.get(z, 0) + 1
    
    return {
        'k': actual_k,
        'zone_adjacency': zone_adjacency,
        'zone_sizes': {str(z): s for z, s in sorted(zone_sizes.items())},
    }


def generate_zones_for_map(map_name: str, data_dir: str = 'data', k: int = None) -> None:
    """특정 맵에 대해 Zone 분할 파일 생성.
    
    Args:
        map_name: 맵 이름 (Anaheim, SiouxFalls 등)
        data_dir: 데이터 디렉토리
        k: Zone 수 (None이면 자동 계산: num_nodes // 14)
    """
    node_file = os.path.join(data_dir, f'{map_name}_node.tntp')
    net_file = os.path.join(data_dir, f'{map_name}_net.tntp')
    
    if not os.path.exists(node_file) or not os.path.exists(net_file):
        print(f"❌ {map_name}: 맵 파일 없음 ({node_file})")
        return
    
    print(f"\n{'='*50}")
    print(f"🗺️  {map_name} Zone 분할 생성")
    print(f"{'='*50}")
    
    # 맵 로드
    dm = DisasterMap(node_file, net_file)
    G = dm.graph
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    
    # K 자동 결정 (Anaheim 기준: 416/14 ≈ 30)
    if k is None:
        k = max(5, num_nodes // 14)
    
    print(f"  노드: {num_nodes}, 엣지: {num_edges}, K: {k}")
    
    # Zone 분할 실행
    node_to_zone = partition_graph_metis(G, k)
    
    # Zone ID를 0부터 연속으로 재매핑 (METIS가 빈 Zone을 만들 수 있으므로)
    unique_zones = sorted(set(node_to_zone.values()))
    zone_remap = {old: new for new, old in enumerate(unique_zones)}
    node_to_zone = {n: zone_remap[z] for n, z in node_to_zone.items()}
    actual_k = len(unique_zones)
    
    # Zone Graph 생성
    zone_graph_data = build_zone_graph(G, node_to_zone, actual_k)
    
    # 파일 저장
    n2z_path = os.path.join(data_dir, f'node_to_zone_{map_name}_k{actual_k}.json')
    zg_path = os.path.join(data_dir, f'zone_graph_{map_name}_k{actual_k}.json')
    
    with open(n2z_path, 'w') as f:
        json.dump({str(n): z for n, z in node_to_zone.items()}, f, indent=2)
    
    with open(zg_path, 'w') as f:
        json.dump(zone_graph_data, f, indent=2)
    
    # 통계 출력
    zone_sizes = list(zone_graph_data['zone_sizes'].values())
    print(f"  실제 Zone 수: {actual_k}")
    print(f"  Zone 크기: min={min(zone_sizes)}, max={max(zone_sizes)}, avg={sum(zone_sizes)/len(zone_sizes):.1f}")
    print(f"  ✅ 저장: {n2z_path}")
    print(f"  ✅ 저장: {zg_path}")


def main():
    parser = argparse.ArgumentParser(description='맵별 Zone 분할 생성')
    parser.add_argument('--map', type=str, default=None,
                        help='특정 맵만 생성 (미지정 시 전체)')
    parser.add_argument('--data', type=str, default='data',
                        help='데이터 디렉토리')
    parser.add_argument('--k', type=int, default=None,
                        help='Zone 수 (미지정 시 자동 계산)')
    args = parser.parse_args()
    
    if args.map:
        maps = [args.map]
    else:
        # data 디렉토리에서 맵 이름 자동 탐색
        maps = []
        for f in os.listdir(args.data):
            if f.endswith('_node.tntp'):
                maps.append(f.replace('_node.tntp', ''))
        maps = sorted(maps)
    
    print(f"대상 맵: {maps}")
    
    for map_name in maps:
        generate_zones_for_map(map_name, args.data, args.k)
    
    print(f"\n🎉 전체 Zone 분할 완료!")


if __name__ == '__main__':
    main()
