
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

class HierarchicalDataset(Dataset):
    def __init__(self, samples, base_data, mode='manager', apsp_matrix=None, max_dist=1.0):
        """
        Args:
            samples: List of dicts (from pickle)
            base_data: PyG Data object (Static Graph)
            mode: 'manager' or 'worker'
            apsp_matrix: [N, N] APSP 행렬 (Worker용, 네트워크 거리 계산)
            max_dist: APSP 정규화용 최대 거리
        """
        self.base_data = base_data
        self.mode = mode
        self.num_nodes = base_data.x.size(0)
        self.apsp_matrix = apsp_matrix  # Worker용: subgoal까지 네트워크 거리
        self.max_dist = max_dist if max_dist > 0 else 1.0
        # 노드 좌표 (방향 벡터 계산용)
        self.node_positions = base_data.x[:, :2]  # [N, 2] (x, y)
        if isinstance(samples, dict):
            # Pre-processed Tensors (Fast Path)
            self.length = len(samples['start_nodes']) if mode == 'manager' else len(samples['curr_nodes'])
            
            if mode == 'manager':
                self.start_nodes = samples['start_nodes']
                self.goal_nodes = samples['goal_nodes']
                self.checkpoint_seqs = samples['checkpoint_seqs']
                self.distance_maps = samples.get('distance_maps', None)  # Soft Label용 거리 맵
            elif mode == 'worker':
                self.curr_nodes = samples['curr_nodes']
                self.target_nodes = samples['target_nodes']
                self.next_hops = samples['next_hops']
                
        else:
            # Legacy List[Dict] (Slow Path, requires conversion)
            self.length = len(samples)
            if mode == 'manager':
                self.start_nodes = torch.tensor([s['start_node'] for s in samples], dtype=torch.long)
                self.goal_nodes = torch.tensor([s['goal_node'] for s in samples], dtype=torch.long)
                self.checkpoint_seqs = [torch.tensor(s['checkpoints'], dtype=torch.long) for s in samples]
            elif mode == 'worker':
                self.curr_nodes = torch.tensor([s['curr'] for s in samples], dtype=torch.long)
                self.target_nodes = torch.tensor([s['target_node'] for s in samples], dtype=torch.long)
                self.next_hops = torch.tensor([s['next_hop'] for s in samples], dtype=torch.long)
                
            # Explicitly delete samples to free memory
            del samples

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Index directly from Tensors
        
        # OPTIMIZATION: Do NOT clone base data blindly.
        # We only need it for concatenation.
        x = self.base_data.x # Reference is enough
        
        if self.mode == 'manager':
            # Input: [x, y, is_origin, is_dest]
            flags = torch.zeros(self.num_nodes, 2)
            # Use scalar item from tensor
            s_node = self.start_nodes[idx].item()
            g_node = self.goal_nodes[idx].item()
            
            flags[s_node, 0] = 1.0
            flags[g_node, 1] = 1.0
            
            x_in = torch.cat([x, flags], dim=1) # [N, 4]
            y = self.checkpoint_seqs[idx] # Tensor sequence
            
            # 거리 맵 반환 (있으면)
            if self.distance_maps is not None:
                dist_map = self.distance_maps[idx]
                return x_in, self.base_data.edge_index, y, dist_map
            
            return x_in, self.base_data.edge_index, y, None
            
        elif self.mode == 'worker':
            # Worker Data is now sequences (List of Tensors for each path)
            # shape of each tensor: [L] where L is sequence length
            c_nodes = self.curr_nodes[idx]
            t_nodes = self.target_nodes[idx]
            n_hops = self.next_hops[idx]
            
            # 여기서 텐서를 묶지 말고 리스트 채로 던져줘서 collate_fn에서 언롤링 단위로 만들도록 함.
            # Base features: node_positions, apsp_matrix (if provided)
            # Computation is deferred to the training loop for Auto-regressive features,
            # or collate_fn handles it. However, to support Autoregressive inference and Teacher Forcing,
            # we need to build dynamic features inside the training loop.
            # Therefore, we just return the raw index sequences.
            return x, self.base_data.edge_index, (c_nodes, t_nodes, n_hops)

def hierarchical_collate(batch):
    # batch is list of (x, edge_index, y) or (x, edge_index, y, dist_map) for Manager
    # y can be Tensor(Seq) or Tensor(1)
    
    data_list = []
    
    # Check if batch has distance_maps (4-tuple)
    has_dist_map = len(batch[0]) == 4
    
    # Check if dataset returns sequence for Manager or tuples for Worker
    sample_y = batch[0][2]
    is_sequence = isinstance(sample_y, torch.Tensor) and sample_y.dim() > 0
    
    if is_sequence:
        # Manager: Pad Sequences
        # Determine Max Len in this Batch
        # Note: Full A* paths can be long (e.g., 50-100 nodes).
        # We dynamic pad to the longest in the batch.
        
        lengths = [item[2].size(0) for item in batch]
        max_len = max(lengths)
        
        # Pad Value: -100 (PyTorch Ignore Index default)
        PAD_VAL = -100 
        
        data_list = []
        ys = []
        dist_maps = []
        
        for item in batch:
            x, edge_index, y = item[0], item[1], item[2]
            dist_map = item[3] if has_dist_map else None
            
            d = Data(x=x, edge_index=edge_index) 
            data_list.append(d)
            ys.append(y)
            if dist_map is not None:
                dist_maps.append(dist_map)
            
        pyg_batch = Batch.from_data_list(data_list)
        
        # Create padded tensor: [Batch, MaxLen]
        padded_y = torch.full((len(ys), max_len), PAD_VAL, dtype=torch.long)
        
        for i, seq in enumerate(ys):
            L = seq.size(0)
            padded_y[i, :L] = seq
            
        pyg_batch.y = padded_y
        
        # Stack distance maps if present
        if dist_maps:
            pyg_batch.dist_maps = torch.stack(dist_maps, dim=0)  # [Batch, num_nodes]
        
        return pyg_batch
        
    else:
        # Worker: y is a tuple of sequences (curr_nodes, target_nodes, next_hops)
        # We don't pad them into a giant batch matrix because we will unroll them.
        # [BugFix] We must construct lists manually and inject them AFTER PyG Batch.from_data_list
        # Otherwise PyG will aggressively concatenate lists of 1D tensors into a single flat tensor list!
        
        c_list, t_list, n_list = [], [], []
        for x, edge_index, (c, t, n) in batch:
            d = Data(x=x, edge_index=edge_index)
            data_list.append(d)
            c_list.append(c)
            t_list.append(t)
            n_list.append(n)
        
        # return single batch for topology
        pyg_batch = Batch.from_data_list(data_list)
        pyg_batch.c_nodes = c_list
        pyg_batch.t_nodes = t_list
        pyg_batch.n_hops = n_list
        
        return pyg_batch
