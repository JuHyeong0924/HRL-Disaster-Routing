import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.worker_env import WorkerEnv
from src.envs.hrl_env import HRLEnv
import torch.nn as nn

class MockWorker(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, xf, aei, edge_attr=None, batch=None, neighbors_mask=None):
        A_N = xf.size(0)
        # logits
        logits = torch.full((A_N,), -1e9, device=xf.device)
        if neighbors_mask is not None:
            # allow neighbor actions
            logits[neighbors_mask > 0] = 1.0
        # If all masked, allow self loops
        # We just return softmax probabilities
        probs = torch.softmax(logits.view(-1, self.num_nodes), dim=-1).view(-1)
        values = torch.zeros(A_N // self.num_nodes, device=xf.device)
        return probs, values, None

def run_verification():
    print("Initializing Environment...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    worker_env = WorkerEnv(
        node_file='data/Anaheim_node.tntp', 
        net_file='data/Anaheim_net.tntp', 
        device=device
    )
    worker = MockWorker(num_nodes=worker_env.num_nodes).to(device)
    hrl_env = HRLEnv(worker=worker, worker_env=worker_env)
    hrl_env.env.disaster_prob = 0.2
    hrl_env.env.dynamic_disaster = True
    
    # 1. Reset
    print("Testing reset()...")
    batch_size = 32
    num_targets = 10
    state_dict = hrl_env.reset(batch_size=batch_size, num_targets=num_targets)
    print(f"State keys: {list(state_dict.keys())}")
    
    # Pre-populate zone sequences to allow calling get_action_mask_batch
    hrl_env.env.zone_sequences = [[0, 1] for _ in range(batch_size)]
    
    # 2. Get action masks
    print("Testing get_action_mask_batch()...")
    masks = hrl_env.env.get_action_mask_batch()
    print(f"Masks shape: {masks.shape}")
    assert masks.shape == (batch_size, hrl_env.env.num_nodes)
    
    # 3. Running multiple steps of step_manager
    print("Testing step_manager speed for 200 steps...")
    t0 = time.time()
    hrl_env.reset(batch_size=batch_size, num_targets=num_targets)
    num_steps = 200
    for step in range(num_steps):
        target_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        zone_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        events, dones = hrl_env.step_manager(target_actions, zone_actions)
        if hrl_env.dones.all():
            print(f"All done at step {step}")
            break
            
    t1 = time.time()
    elapsed = t1 - t0
    print(f"Finished {step+1} steps of HRL environment step_manager.")
    print(f"Total time elapsed: {elapsed:.4f} seconds")
    print(f"Average time per step: {elapsed / (step + 1):.4f} seconds")
    
    print("\n✅ Verification SUCCESSFUL! All optimized code runs correctly without crashes.")

if __name__ == '__main__':
    run_verification()
