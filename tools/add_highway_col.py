
import os
import networkx as nx

def add_highway_column(map_name="Anaheim"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, "data", f"{map_name}_net.tntp")
    
    print(f"🛣️ Processing {net_file}...")
    
    with open(net_file, 'r') as f:
        lines = f.readlines()
        
    start_line = 0
    header_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('~'):
            start_line = i + 1
            header_line = i
            break
            
    # Parse Edges
    edges = []
    # (u, v, speed, line_idx)
    
    G_speed = nx.Graph()
    
    for i in range(start_line, len(lines)):
        line = lines[i]
        parts = line.strip().replace(';', '').split()
        if len(parts) < 3: continue
        
        u = int(parts[0])
        v = int(parts[1])
        
        # Parse Speed (Index 7 check)
        # Columns: Init, Term, Cap, Len, FFT, B, Power, Speed, Toll, Type
        # Index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        speed = 0.0
        if len(parts) > 7:
            speed = float(parts[7])
            
        edges.append({
            'u': u, 'v': v, 'speed': speed, 'line_idx': i, 'original': line
        })
        
        if speed >= 4842:
            G_speed.add_edge(u, v, index=i)
            
    # Connectivity Filter
    # Keep components with size >= 5
    core_highway_indices = set()
    core_highway_nodes = set()
    
    print(f"🔍 Found {G_speed.number_of_edges()} high-speed edges (>= 4842). Checking connectivity...")
    
    comps = list(nx.connected_components(G_speed))
    valid_count = 0
    
    for comp in comps:
        # Get subgraph to count edges
        subg = G_speed.subgraph(comp)
        if subg.number_of_edges() >= 5:
            valid_count += subg.number_of_edges()
            for u, v, d in subg.edges(data=True):
                core_highway_indices.add(d['index'])
            # Collect Core Nodes
            core_highway_nodes.update(comp)
                
    print(f"✅ Identified {valid_count} core highway edges (Component Size >= 5).")
    print(f"   Core Highway Nodes: {len(core_highway_nodes)}")
    
    # Ramp Expansion (1-Hop Neighbor Edges)
    # Include ANY edge that touches a Core Highway Node
    final_highway_indices = set(core_highway_indices)
    ramp_count = 0
    
    for edge in edges:
        if edge['line_idx'] in core_highway_indices: continue
        
        u, v = edge['u'], edge['v']
        if u in core_highway_nodes or v in core_highway_nodes:
            final_highway_indices.add(edge['line_idx'])
            ramp_count += 1
            
    print(f"🔗 Added {ramp_count} Ramp/Connector edges (Connected to Core Nodes).")
    print(f"🛣️ Total Highway System Edges: {len(final_highway_indices)}")

    
    # Update Header
    # ~	init_node	term_node	capacity	length	free_flow_time	b	power	speed	toll	link_type	;
    old_header = lines[header_line].strip()
    if 'is_highway' not in old_header:
        # Remove trailing ; if exists, add is_highway, then ;
        clean_header = old_header.replace(';', '').strip()
        new_header = clean_header + "\tis_highway\t;\n"
        lines[header_line] = new_header
    else:
        print("⚠️ 'is_highway' column already exists in header.")
        
    # Update Lines
    for i in range(start_line, len(lines)):
        line = lines[i].strip()
        if not line: continue
        
        # Check if already has column?
        parts = line.replace(';', '').split()
        
        is_h = 1 if i in final_highway_indices else 0
        
        # Reconstruct line
        # Assuming tab separated original
        # We append the value.
        
        # Carefully handle existing format
        original_content = lines[i].replace(';', '').rstrip()
        
        # If we re-run, avoid appending again if logic allows?
        # Simpler: Just append \t1\t;
        # But if it already exists?
        # Let's assume we are appending strictly. 
        # CAUTION: If user ran this before, we might duplicate.
        # Check column count
        # Original Anaheim cols = 10.
        # If parts > 10, maybe overwrite?
        
        if len(parts) > 10:
             # Look for highway col? 
             # Let's just reconstruct based on first 10 columns + new
             base_parts = parts[:10]
             new_line = "\t" + "\t".join(base_parts) + f"\t{is_h}\t;\n"
        else:
             new_line = original_content + f"\t{is_h}\t;\n"
             
        lines[i] = new_line
        
    # Save
    with open(net_file, 'w') as f:
        f.writelines(lines)
        
    print(f"💾 Updated {net_file} with 'is_highway' column.")

if __name__ == "__main__":
    add_highway_column()
