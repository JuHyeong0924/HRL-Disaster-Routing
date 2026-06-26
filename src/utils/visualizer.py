import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import imageio

class DisasterVisualizer:
    def __init__(self, dm, n2z):
        """
        dm: DisasterMap instance
        n2z: Dict mapping node -> zone_idx
        """
        self.dm = dm
        self.G = dm.graph
        self.pos = dm.pos
        self.n2z = n2z
        # Grid-based zone cells (matching evaluate.py)
        all_x = [p[0] for p in self.pos.values()]
        all_y = [p[1] for p in self.pos.values()]
        self.min_x, max_x = min(all_x), max(all_x)
        self.min_y, max_y = min(all_y), max(all_y)
        
        import math
        self.N_grid = math.ceil(math.sqrt(len(self.pos) / 16.0))
        self.N_grid = max(2, self.N_grid)
        eps_x = (max_x - self.min_x) * 1e-6
        eps_y = (max_y - self.min_y) * 1e-6
        max_x += eps_x
        max_y += eps_y
        self.dx = (max_x - self.min_x) / self.N_grid
        self.dy = (max_y - self.min_y) / self.N_grid

        from collections import defaultdict
        self.zone_to_cells = defaultdict(list)
        for node in list(self.G.nodes()):
            x, y = self.pos[node]
            gx = int((x - self.min_x) / self.dx)
            gy = int((y - self.min_y) / self.dy)
            gx = min(max(gx, 0), self.N_grid-1)
            gy = min(max(gy, 0), self.N_grid-1)
            z = self.n2z.get(node)
            if z is not None:
                self.zone_to_cells[z].append((gx, gy))

    def draw_grid_zone(self, ax, z_idx, color, draw_text=False, step_idx=None):
        if z_idx not in self.zone_to_cells: return None
        cells = set(self.zone_to_cells[z_idx])
        sum_cx, sum_cy = 0, 0
        for gx, gy in cells:
            cx, cy = self.min_x + gx * self.dx, self.min_y + gy * self.dy
            rect = patches.Rectangle((cx, cy), self.dx, self.dy, linewidth=0, facecolor=color, alpha=0.5, zorder=0)
            ax.add_patch(rect)
            sum_cx += cx + self.dx/2
            sum_cy += cy + self.dy/2
            
        center_pt = (sum_cx / len(cells), sum_cy / len(cells))
        if draw_text and step_idx is not None:
            ax.text(center_pt[0], center_pt[1], str(step_idx), color='black', fontsize=14, weight='bold', 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        return center_pt

    def plot_state(self, hrl_env, step_idx, save_dir, filename_prefix="step", trajectory=None, mission_zone=None, zone_sequence=None, worker_edge=None, worker_progress=1.0, global_time=None):
        """
        현재 환경의 상태를 matplotlib으로 시각화하여 PNG로 저장합니다.
        - global_time: 0.1 단위 틱 기반 시뮬레이션의 글로벌 시간 (None이면 hrl_env.current_time 사용)
        - worker_edge: (u, v) 워커가 이동 중인 간선
        - worker_progress: 0.0 ~ 1.0 간선 이동 진행률
        """
        os.makedirs(save_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_title(f"Disaster Routing HRL - Step {step_idx}", fontsize=16)
        
        # 1. Background Nodes and Edges (Light Gray)
        nodes = list(self.G.nodes())
        
        nx.draw_networkx_nodes(
            self.G, self.pos, nodelist=nodes, node_color='lightgray', 
            node_size=10, alpha=0.5, ax=ax
        )
        
        # 2. Draw Edges and Damages
        normal_edges = []
        minor_damaged_edges = []
        severe_damaged_edges = []
        minor_damage_values = []
        
        for u, v, d in self.G.edges(data=True):
            dmg = d.get('damage', 0.0)
            if dmg > 0.8:
                severe_damaged_edges.append((u, v))
            elif dmg > 0.05:
                minor_damaged_edges.append((u, v))
                minor_damage_values.append(dmg)
            else:
                normal_edges.append((u, v))
                
        # Normal edges
        nx.draw_networkx_edges(
            self.G, self.pos, edgelist=normal_edges, 
            edge_color='lightgray', width=0.5, alpha=0.5, ax=ax
        )
        
        # Minor damaged edges (Gradient colors, dashed lines)
        if minor_damaged_edges:
            cmap = plt.get_cmap('YlOrRd')
            norm = mcolors.Normalize(vmin=0, vmax=0.8)
            edge_colors = [cmap(norm(val)) for val in minor_damage_values]
            
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=minor_damaged_edges, 
                edge_color=edge_colors, width=2.5, style='dashed', ax=ax
            )
            
        # Severe damaged edges (Solid Red, Thick)
        if severe_damaged_edges:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=severe_damaged_edges, 
                edge_color='red', width=5.0, style='solid', ax=ax
            )
            # Add an X marker in the middle of severe edges
            for u, v in severe_damaged_edges:
                x = (self.pos[u][0] + self.pos[v][0]) / 2
                y = (self.pos[u][1] + self.pos[v][1]) / 2
                ax.plot(x, y, marker='X', color='darkred', markersize=8)
                
        # 3. Draw Grid Zones
        b = 0 # Batch 0만 시각화
        curr_idx = int(hrl_env.env.curr_nodes[b].item())
        curr_node = hrl_env.env.idx_to_node[curr_idx]
        curr_zone = self.n2z.get(curr_node)
        
        # Draw Current Zone (초록색)
        if curr_zone is not None:
            self.draw_grid_zone(ax, curr_zone, color='lightgreen')
            
        # Draw Zone Sequence (Path Zones, 하늘색)
        if zone_sequence is not None:
            for z in zone_sequence:
                if z != curr_zone and z != mission_zone:
                    self.draw_grid_zone(ax, z, color='lightblue')
                    
        # Draw Mission Zone (빨간색)
        if mission_zone is not None and mission_zone != curr_zone:
            self.draw_grid_zone(ax, mission_zone, color='lightcoral')

        # 4. Draw Trajectory (Worker's path)
        if trajectory and len(trajectory) > 1:
            path_edges = [(trajectory[i], trajectory[i+1]) for i in range(len(trajectory)-1)]
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=path_edges, 
                edge_color='black', width=2.5, ax=ax
            )

        # 5. Draw Targets
        num_targets = hrl_env.num_targets
        
        for i in range(num_targets):
            t_idx = int(hrl_env.targets[b, i].item())
            t_node = hrl_env.env.idx_to_node[t_idx]
            deadline = int(hrl_env.deadlines[b, i].item())
            
            if hrl_env.target_rescued[b, i]:
                # Rescued targets
                nx.draw_networkx_nodes(
                    self.G, self.pos, nodelist=[t_node], node_shape='o', 
                    node_color='lightgray', node_size=50, alpha=0.5, ax=ax
                )
            elif hasattr(hrl_env, 'target_failed') and hrl_env.target_failed[b, i]:
                # Failed targets (Deadline exceeded) -> Do not display (or just draw as a small black cross, but user asked to remove them)
                continue
            else:
                # Active targets
                nx.draw_networkx_nodes(
                    self.G, self.pos, nodelist=[t_node], node_shape='o', 
                    node_color='red', node_size=150, ax=ax
                )
                # Annotate Deadline
                x, y = self.pos[t_node]
                ax.annotate(f"TW: {deadline}", 
                            xy=(x, y), xytext=(0, 15), textcoords='offset points',
                            ha='center', va='bottom',
                            color='darkred', fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
        # 6. Draw Current Worker Position
        if worker_edge is not None:
            u, v = worker_edge
            ux, uy = self.pos[u]
            vx, vy = self.pos[v]
            wx = ux + (vx - ux) * worker_progress
            wy = uy + (vy - uy) * worker_progress
            ax.plot(wx, wy, marker='X', color='green', markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        else:
            nx.draw_networkx_nodes(
                self.G, self.pos, nodelist=[curr_node], node_shape='X', 
                node_color='green', node_size=150, ax=ax
            )

        # Info text
        rescued_count = int(hrl_env.target_rescued[b].sum().item())
        time_elapsed = global_time if global_time is not None else hrl_env.current_time[b].item()
        info_str = f"Rescued: {rescued_count}/{num_targets} | Time: {time_elapsed:.1f}/{hrl_env.max_time}"
        ax.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

        ax.axis('off')
        
        filepath = os.path.join(save_dir, f"{filename_prefix}_{step_idx:04d}.png")
        plt.savefig(filepath, dpi=100)
        plt.close(fig)
        return filepath

    @staticmethod
    def create_gif(image_paths, output_path, fps=5):
        """PNG 이미지 목록을 GIF로 묶어줍니다."""
        images = []
        for path in image_paths:
            images.append(imageio.imread(path))
        imageio.mimsave(output_path, images, fps=fps)
        print(f"✅ GIF saved to {output_path}")
