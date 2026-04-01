
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import csv
import random
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.disaster_env import DisasterEnv

def visualize_physics(map_name):
    print(f"🧪 Starting Physics Engine Verification for {map_name}...")
    
    # Files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    node_file = os.path.join(base_dir, "data", f"{map_name}_node.tntp")
    net_file = os.path.join(base_dir, "data", f"{map_name}_net.tntp")
    
    if not os.path.exists(node_file):
        print(f"❌ File not found: {node_file}")
        return

    # Create Output Directory
    # User Request: tests/figs/physics/{MapName}/ -> tests/physics/{MapName}/
    test_dir = os.path.join(base_dir, "tests")
    # DIRECTORY UPDATE: Remove "figs" from path, use "physics" directly
    base_output_dir = os.path.join(test_dir, "physics", map_name)
    
    # Create Sub-directories
    log_dir = os.path.join(base_output_dir, "logs")
    img_dir = os.path.join(base_output_dir, "images")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # print(f"📂 Output Directories Created:\n   Logs: {log_dir}\n   Images: {img_dir}")

    # Initialize Env (Batch=1)
    env = DisasterEnv(node_file, net_file)
    env.reset(batch_size=1)
    
    # Access Internal Physics State
    # damage_states: [Batch, NumEdges] (0:Normal, 1:Damaged, 2:Collapsed) -- For RL
    # pga: [Batch, NumEdges] -- For HAZUS Visualization
    
    # 0. Verify Elevated Roads (User Request)
    plot_elevated_check(env, img_dir)
    
    # 4. Generate & Visualize Main Shock
    print(f"⚠️ Triggering Main Shock (Stochastic Network Event)...")
    base_states = env.reset(batch_size=1)
    
    # Extract Intensity Score
    # env.damage_states is updated in reset()
    # We need to peek at 'pga' (Score) which is local in reset.
    # But wait, reset() doesn't return pga.
    # We need to modify reset slightly or just rely on state?
    # Actually env.last_pga or similar would be good, but for now we can't get it easily without mod.
    # Wait, let's just use empty pga for log if needed, OR better:
    # Just save the states. Log expects pga_tensor.
    
    # We will modify reset to save last_score for visualization convenience
    # But for now, let's just log the STATES.
    
    # Actually, let's trust that the user just wants the final state.
    # But to log "Intensity", we need the score.
    # Let's add 'self.last_score' to env in the next step if needed.
    # For now, let's assume env.last_score exists (I will add it to env).
    
    if hasattr(env, 'last_score'):
        main_score = env.last_score
    else:
        main_score = torch.zeros(1, env.num_physical_edges)
        
    plot_damage_map(env, main_score[0].cpu().numpy(), img_dir, title="Shock_0_Intensity")
    
    # 3. Plot State (Actual vs Theoretical)
    # 3a. Actual
    plot_damage_state_map(env, env.damage_states[0].cpu().numpy(), img_dir, title="Shock_0_Damage_Actual")
    save_damage_log(env, main_score, log_dir, "Shock_0_Damage_Actual", state_tensor=env.damage_states, event_label="Shock_0_Actual")
    
    # 3b. Theoretical
    if hasattr(env, 'damage_states_theoretical'):
        theo_state = env.damage_states_theoretical[0].cpu().numpy()
        plot_damage_state_map(env, theo_state, img_dir, title="Shock_0_Damage_Theoretical")
        save_damage_log(env, main_score, log_dir, "Shock_0_Damage_Theoretical", 
                        state_tensor=env.damage_states_theoretical, event_label="Shock_0_Theoretical")
    # Track Max PGA for Cumulative Visualization (User Request: 5-level like Intensity)
    cumulative_pga = main_score[0].cpu().numpy().copy() if isinstance(main_score[0].cpu().numpy(), np.ndarray) else main_score[0].cpu().numpy().clone()
    
    # 2. Visualize Scheduled Aftershocks
    print("⚠️ Simulating Seismic Schedule...")
    
    # Sort steps excluding 0 (Main Shock)
    sorted_steps = sorted([t for t in env.seismic_schedule if t > 0])
    print(f"   Scheduled Events at steps: {sorted_steps}")
    
    for i, t in enumerate(sorted_steps):
        params = env.seismic_schedule[t]
        
        # Force Time Sync (Mock Simulator Time)
        # Step increment happened inside step(), so we set to t-1 to hit t on increment
        env.step_count[:] = t - 1
        
        # Execute Step (Triggers Physics via Schedule)
        # Dummy Action: Stay (0)
        dummy_action = torch.zeros(1, dtype=torch.long, device=env.device)
        
        # Step
        # This will trigger the scheduled event logic we added to step()
        env.step(dummy_action)
        
        print(f"   [Event #{i+1} @ t={t}] {params['type'].upper()} Triggered. Intensity: {params['intensity_min']}~{params['intensity_max']}")
        
        # Visualize Result
        
        # Note: We need access to the Intensity generated in this step for visualization.
        # But step() accumulates damage and doesn't return the raw PGA of the event.
        # However, env.last_score holds the LAST generated PGA.
        if hasattr(env, 'last_score'):
            as_pga = env.last_score
        else:
            as_pga = torch.zeros(1, env.num_physical_edges)
            
        # 1. Plot Intensity (Incremental)
        pga_after = as_pga[0].cpu().numpy()
        title_int = f"Shock_{i+1}_Intensity"
        plot_damage_map(env, pga_after, img_dir, title=title_int)
        
        # 2. Cumulative Intensity Calculation
        cumulative_pga = np.maximum(cumulative_pga, pga_after)
        
        # [User Request] HAZUS Scale Cumulative Map
        plot_damage_map(env, cumulative_pga, img_dir, title=f"Shock_{i+1}_Cumulative_Intensity_HAZUS")
        
        # 3. Plot Cumulative State (Actual vs Theoretical)
        # 3a. Actual (With Noise)
        curr_states = env.damage_states[0].cpu().numpy()
        title_state_act = f"Shock_{i+1}_Cumulative_State_Actual"
        plot_damage_state_map(env, curr_states, img_dir, title=title_state_act)
        
        # 3b. Theoretical (No Noise - HAZUS Only)
        if hasattr(env, 'damage_states_theoretical'):
            curr_states_theo = env.damage_states_theoretical[0].cpu().numpy()
            title_state_theo = f"Shock_{i+1}_Cumulative_State_Theoretical"
            plot_damage_state_map(env, curr_states_theo, img_dir, title=title_state_theo)
        
        # 4. Save Logs
        # Event Specific Log
        save_damage_log(env, as_pga, log_dir, title_int, state_tensor=None, event_label=f"Shock_{i+1}")
        
        # Cumulative Log (Actual)
        save_damage_log(env, torch.tensor(cumulative_pga).unsqueeze(0), log_dir, title_state_act, 
                        state_tensor=env.damage_states, event_label="Cumulative_Actual")
                        
        # Cumulative Log (Theoretical)
        if hasattr(env, 'damage_states_theoretical'):
             save_damage_log(env, torch.tensor(cumulative_pga).unsqueeze(0), log_dir, title_state_theo, 
                        state_tensor=env.damage_states_theoretical, event_label="Cumulative_Theoretical")
        
    print(f"✅ Visualization Complete.")

def save_damage_log(env, pga_tensor, output_dir, event_name, state_tensor=None, event_label="Event"):
    """
    Saves a CSV log containing:
    - Link Info: U, V, Speed, Is_Highway, Score, Damage_State
    """
    log_path = os.path.join(output_dir, f"{event_name}_Log.csv")
    
    # Edges
    G = env.map_core.graph
    edge_list = list(G.edges(data=True))
    
    # Attributes
    if hasattr(env, 'edge_speeds'):
        speeds = env.edge_speeds.cpu().numpy()
    else:
        speeds = [0.0] * len(edge_list)
        
    if hasattr(env, 'edge_is_highways'):
        is_highways = env.edge_is_highways.cpu().numpy()
    else:
        is_highways = [0.0] * len(edge_list)
        
    # PGA [Batch, Edges] -> [Edges] (Handle Tensor/Numpy)
    if isinstance(pga_tensor, torch.Tensor):
        if len(pga_tensor.shape) > 1:
            pga_vals = pga_tensor[0].cpu().numpy()
        else:
            pga_vals = pga_tensor.cpu().numpy()
    else:
        pga_vals = pga_tensor
        
    # State (Accumulated)
    state_vals = None
    if state_tensor is not None:
        if isinstance(state_tensor, torch.Tensor):
            if len(state_tensor.shape) > 1:
                state_vals = state_tensor[0].cpu().numpy()
            else:
                state_vals = state_tensor.cpu().numpy()
        else:
            state_vals = state_tensor
        
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["# Event_Label", event_label])
        writer.writerow([])
        
        # Columns
        writer.writerow(["Edge_Index", "U_Node", "V_Node", "Speed", "Is_Highway", "Intensity_Score", "HAZUS_Scale", "Damage_Desc", "Damage_Code"])
        
        for i, (u, v, d) in enumerate(edge_list):
            if i >= len(pga_vals): break
            
            pga = pga_vals[i]
            spd = speeds[i] if i < len(speeds) else 0
            is_h = is_highways[i] if i < len(is_highways) else 0
            
            # HAZUS Scale (Intensity Score System 0-10)
            if pga < 2.0: hazus = "Normal"
            elif pga < 3.0: hazus = "Slight"
            elif pga < 4.0: hazus = "Moderate"
            elif pga < 5.0: hazus = "Extensive"
            else: hazus = "Complete"
            
            # Determine Description (Physical State)
            if state_vals is not None and i < len(state_vals):
                code = int(state_vals[i])
                if code == 0: desc = "Normal"
                else: desc = "Damaged"
            else:
                # Fallback to Score estimate (Only for Main Shock initial check if state not passed)
                code = -1
                desc = "Est_Normal"
                # Game Physics Thresholds (approx avg)
                # Bridges Collapse > 5.0, Roads > 7.5 -> Avg ~6.0?
                # Let's use conservative estimates for the log description
                if pga > 5.5: desc = "Est_Complete"
                elif pga > 3.0: desc = "Est_Damaged"
            
            writer.writerow([i, u, v, spd, int(is_h), f"{pga:.4f}", hazus, desc, code])
            
    print(f"📝 Saved Log: {log_path}")

def plot_damage_state_map(env, state_data, output_dir, title="Damage State Map"):
    """
    Visualizes DISCRETE damage states (0: Normal, 1: Damaged, 2: Collapsed).
    Matches the CSV Log 'Damage_Code'.
    """
    G = env.map_core.graph
    pos = env.map_core.pos
    
    plt.figure(figsize=(20, 16))
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='#dddddd', edgecolors='grey')
    
    edge_list = list(G.edges())
    
    # State Lists
    edges_normal = []
    edges_damaged = []
    edges_collapsed = []
    
    state_arr = state_data.flatten() if hasattr(state_data, 'flatten') else state_data
    
    for i, (u, v) in enumerate(edge_list):
        if i >= len(state_arr): break
        val = int(state_arr[i])
        
        if val == 0:
            edges_normal.append((u, v))
        else: # val == 1
            edges_damaged.append((u, v))
            
    # Draw Edges
    # Normal: Grey
    nx.draw_networkx_edges(G, pos, edgelist=edges_normal, width=1.5, edge_color='#cccccc', alpha=0.4)
    # Damaged: Red (Blocked)
    nx.draw_networkx_edges(G, pos, edgelist=edges_damaged, width=4.0, edge_color='#D32F2F', alpha=1.0)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='#cccccc', lw=1.5, label=f'Normal (0) - {len(edges_normal)}'),
        plt.Line2D([0], [0], color='#D32F2F', lw=4.0, label=f'Damaged (1) - {len(edges_damaged)}'),
    ]
    
    plt.title(title + " (Probabilistic State)", fontsize=20, fontweight='bold', pad=20)
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📸 Saved state visualization: {save_path}")
    plt.close()

def plot_damage_map(env, pga_data, output_dir, title="Damage Map"):
    G = env.map_core.graph
    pos = env.map_core.pos
    
    # [Style Update] Matching visualize_result.py style
    plt.figure(figsize=(20, 16))
    ax = plt.gca()
    
    # Draw Nodes: Size 30, Color #dddddd, Edge Grey
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='#dddddd', edgecolors='grey')
    
    # HAZUS 5-Level Classification based on PGA (g)
    # Thresholds:
    # < 0.10g: Normal (Gray)
    # 0.10 ~ 0.25g: Slight (Yellow)
    # 0.25 ~ 0.40g: Moderate (Orange)
    # 0.40 ~ 0.80g: Extensive (Dark Orange/Red-Orange)
    # >= 0.80g: Complete (Red)
    
    edge_list = list(G.edges())
    # Note: env.num_physical_edges should match len(edge_list) if properly synced
    # We map pga[i] to edge_list[i].
    
    # Lists for drawing
    edges_normal = []
    edges_slight = []
    edges_moderate = []
    edges_extensive = []
    edges_complete = []
    
    for i, (u, v) in enumerate(edge_list):
        if i >= len(pga_data): break
        val = pga_data[i]
        
        if val < 2.0:
            edges_normal.append((u, v))
        elif val < 3.0:
            edges_slight.append((u, v))
        elif val < 4.0:
            edges_moderate.append((u, v))
        elif val < 5.0:
            edges_extensive.append((u, v))
        else:
            edges_complete.append((u, v))
            
    # Draw Edges (Layered)
    # Width and Alpha tuning to match 'Clean' look
    # Normal: Thinner, less obtrusive
    nx.draw_networkx_edges(G, pos, edgelist=edges_normal, width=1.5, edge_color='#cccccc', alpha=0.4)
    
    # Damage Levels: Progressive width and opacity
    nx.draw_networkx_edges(G, pos, edgelist=edges_slight, width=2.0, edge_color='#FFEB3B', alpha=0.7) # Yellow
    nx.draw_networkx_edges(G, pos, edgelist=edges_moderate, width=3.0, edge_color='#FF9800', alpha=0.8) # Orange
    nx.draw_networkx_edges(G, pos, edgelist=edges_extensive, width=4.0, edge_color='#FF5722', alpha=0.9) # Deep Orange
    nx.draw_networkx_edges(G, pos, edgelist=edges_complete, width=5.0, edge_color='#500000', alpha=1.0) # Very Dark Red (Complete)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='#cccccc', lw=1.5, label=f'Normal (<0.1g) - {len(edges_normal)}'),
        plt.Line2D([0], [0], color='#FFEB3B', lw=2.0, label=f'Slight (0.1~0.25g) - {len(edges_slight)}'),
        plt.Line2D([0], [0], color='#FF9800', lw=3.0, label=f'Moderate (0.25~0.4g) - {len(edges_moderate)}'),
        plt.Line2D([0], [0], color='#FF5722', lw=4.0, label=f'Extensive (0.4~0.8g) - {len(edges_extensive)}'),
        plt.Line2D([0], [0], color='#500000', lw=5.0, label=f'Complete (≥0.8g) - {len(edges_complete)}'),
    ]
    
    plt.title(title + " (HAZUS Scale)", fontsize=20, fontweight='bold', pad=20)
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📸 Saved visualization: {save_path}")
    plt.close()

def plot_elevated_check(env, output_dir):
    """
    Visualizes Elevated Roads (is_highway == 1) in Cyan to verify identification logic.
    """
    print("🔍 Generating Elevated Road Verification Map...")
    G = env.map_core.graph
    pos = env.map_core.pos
    
    # Validation: Check if we have edge_is_highways
    use_highway_attr = hasattr(env, 'edge_is_highways')
    if not use_highway_attr and not hasattr(env, 'edge_speeds'):
        print("❌ No edge attributes found. Cannot verify elevated roads.")
        return

    plt.figure(figsize=(20, 16))
    
    # Draw Nodes (Subtle)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='#eeeeee', edgecolors='grey', alpha=0.5)
    
    edge_list = list(G.edges())
    
    if use_highway_attr:
        values = env.edge_is_highways.cpu().numpy()
        threshold = 0.5
        label_key = "Is Highway"
    else:
        values = env.edge_speeds.cpu().numpy()
        threshold = 4842
        label_key = "Speed >= 4842"
    
    edges_elevated = []
    edges_normal = []
    
    for i, (u, v) in enumerate(edge_list):
        if i >= len(values): break
        val = values[i]
        
        if val >= threshold:
            edges_elevated.append((u, v))
        else:
            edges_normal.append((u, v))
            
    # Draw Normal
    nx.draw_networkx_edges(G, pos, edgelist=edges_normal, width=1.5, edge_color='#cccccc', alpha=0.3)
    
    # Draw Elevated (Cyan, Thicker)
    nx.draw_networkx_edges(G, pos, edgelist=edges_elevated, width=4.0, edge_color='cyan', alpha=1.0)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='#cccccc', lw=1.5, label=f'Normal Road - Count: {len(edges_normal)}'),
        plt.Line2D([0], [0], color='cyan', lw=4.0, label=f'Elevated Road ({label_key}) - Count: {len(edges_elevated)}'),
    ]
    
    plt.title(f"Elevated Road Verification (Source: {label_key})", fontsize=20, fontweight='bold', pad=20)
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "Elevated_Road_Verification.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📸 Saved verification map: {save_path}")
    plt.close()

if __name__ == "__main__":
    # [User Request] 무조건 Anaheim으로만 작동
    map_name = "Anaheim"
    visualize_physics(map_name)
