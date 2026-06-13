import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.manager_env import ManagerEnv
from src.envs.worker_env import WorkerEnv
from src.models.manager import Manager
from src.models.worker import Worker

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing Worker...")
    worker = Worker(node_dim=4, hidden_dim=64, num_layers=2).to(device)
    worker_env = WorkerEnv(
        "data/Anaheim_node.tntp",
        "data/Anaheim_net.tntp",
        zone_json="data/grid_Anaheim_node_to_zone.json",
        zone_graph_json="data/grid_Anaheim_zone_graph.json"
    )
    s = worker_env.reset(batch_size=2)
    print("Worker state shape:", s.shape)
    
    print("Testing Manager...")
    manager = Manager(node_dim=7, hidden_dim=64, num_layers=2).to(device)
    manager_env = ManagerEnv(
        node_file="data/Anaheim_node.tntp",
        net_file="data/Anaheim_net.tntp",
        worker=worker,
        zone_json="data/grid_Anaheim_node_to_zone.json",
        zone_graph_json="data/grid_Anaheim_zone_graph.json",
        device=device
    )
    manager_env.reset()
    state = manager_env.get_manager_state()
    print("Manager state shape:", state.shape)
    print("Test passed successfully!")

if __name__ == "__main__":
    main()
