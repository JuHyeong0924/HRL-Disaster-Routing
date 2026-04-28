import json
import networkx as nx

def build_zone_graph(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    for z in range(data['k']):
        G.add_node(z)
        
    for z, neighbors in data['zone_adjacency'].items():
        z = int(z)
        for n in neighbors:
            # 기본 가중치 1.0 (향후 Manager RL 출력값으로 대체)
            G.add_edge(z, n, weight=1.0)
    return G

G = build_zone_graph('data/zone_graph_k30.json')
path = nx.astar_path(G, 0, 10, weight='weight')
print("A* Path 0 -> 10:", path)
