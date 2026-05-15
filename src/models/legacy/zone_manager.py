import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class ZoneManager(nn.Module):
    """
    Phase 2: Zone-Level Manager (신규 설계)
    - 30개의 Zone 위상 정보만을 입력받아 구역 간 연결 가중치(Cost/Risk)를 추론.
    - 이후 A* 알고리즘과 결합되어 최단 안전 경로를 생성하는 데 사용됨.
    """
    def __init__(self, node_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.1):
        super(ZoneManager, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Zone Graph Embedding (Local Topology)
        self.gcn1 = GCNConv(node_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # 2. Edge Evaluator (Risk 추론기)
        # 출발 Zone과 도착 Zone의 임베딩을 받아서 해당 Edge의 Cost 추론
        self.edge_evaluator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # 0.0 ~ 1.0 (Risk score)
        )
        
    def forward(self, x, edge_index):
        """
        Args:
            x: [N_zones, node_dim] (Zone Node Features)
            edge_index: [2, E_zones] (Zone Connectivity)
        Returns:
            edge_costs: [E_zones] (Risk score 0.0 ~ 1.0)
        """
        # Node Embedding
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        
        # Edge Evaluation
        # edge_index: [2, E] -> src, dst
        src_emb = h[edge_index[0]] # [E, H]
        dst_emb = h[edge_index[1]] # [E, H]
        
        edge_input = torch.cat([src_emb, dst_emb], dim=-1) # [E, 2H]
        edge_costs = self.edge_evaluator(edge_input).squeeze(-1) # [E]
        
        return edge_costs
