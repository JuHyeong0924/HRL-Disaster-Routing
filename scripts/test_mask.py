import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.eval_utils import get_manager_action

# We will monkey-patch get_manager_action
original_get_manager_action = get_manager_action

def patched_get_manager_action(manager, hrl_env, device):
    # This is a bit tricky because zam is fetched inside get_manager_action.
    # So we'll patch manager.get_zone_logits instead!
    original_get_zone_logits = manager.get_zone_logits
    
    def wrapped_get_zone_logits(query, selected_target_emb, zone_embeddings, zone_adj_mask, zone_batch, selected_target_zone_idx, zone_dist_matrix):
        # zone_adj_mask: [B, K]
        B = zone_adj_mask.size(0)
        c_nodes = hrl_env.env.curr_nodes
        for b in range(B):
            c_node = hrl_env.env.idx_to_node[int(c_nodes[b])]
            c_zone = hrl_env.env.n2z[c_node]
            if selected_target_zone_idx[b] != c_zone:
                zone_adj_mask[b, c_zone] = 0
                
        return original_get_zone_logits(query, selected_target_emb, zone_embeddings, zone_adj_mask, zone_batch, selected_target_zone_idx, zone_dist_matrix)
        
    manager.get_zone_logits = wrapped_get_zone_logits
    try:
        res = original_get_manager_action(manager, hrl_env, device)
    finally:
        manager.get_zone_logits = original_get_zone_logits
    return res

import src.utils.eval_utils as eval_utils
eval_utils.get_manager_action = patched_get_manager_action

# Now run evaluation
os.system(f"python scripts/evaluate.py --mode benchmark --episodes 5 --worker_ckpt logs/rl_worker_stage/2026-06-17_082247_worker/best.pt --manager_ckpt logs/rl_manager_stage/2026-06-17_114459_manager/best_manager.pt")
