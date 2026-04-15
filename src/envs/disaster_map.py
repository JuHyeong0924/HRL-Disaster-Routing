import networkx as nx
import pandas as pd
import random
import os

class DisasterMap:
    def __init__(self, node_file, net_file):
        """
        SiouxFalls 데이터셋을 로드하여 NetworkX 그래프로 구축
        """
        self.node_file = node_file
        self.net_file = net_file
        self.graph = nx.Graph() # 무방향 그래프
        self.pos = {} # 시각화용 좌표 저장소
        
        # [수정] Converter에서 사용할 맵의 경계값 (Max X, Max Y)
        self.bounds = (0, 0) 
        
        # 데이터 로드 및 그래프 구축
        self._load_network()
        
    def _load_network(self):
        # 1. Node 데이터 로드 (좌표 포함)
        all_x = [] # 좌표 범위 계산용 리스트
        all_y = []
        
        try:
            with open(self.node_file, 'r') as f:
                lines = f.readlines()
            
            start_line = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('Node'):
                    start_line = i
                    break
            
            for line in lines[start_line+1:]:
                parts = line.strip().replace(';', '').split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    
                    self.graph.add_node(node_id, x=x, y=y)
                    self.pos[node_id] = (x, y)
                    
                    # 범위 계산을 위해 수집
                    all_x.append(x)
                    all_y.append(y)
            
            # [핵심 수정] 맵의 경계값 계산 (Min-Max Normalization용)
            if all_x and all_y:
                self.bounds = (min(all_x), min(all_y), max(all_x), max(all_y))
            else:
                self.bounds = (0.0, 0.0, 1.0, 1.0) # 기본값 (예외 처리)

        except Exception as e:
            print(f"❌ Error loading node file: {e}")
            raise e

        # 2. Link(Edge) 데이터 로드
        try:
            with open(self.net_file, 'r') as f:
                lines = f.readlines()
                
            start_line = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('~'):
                    start_line = i + 1
                    break
            
            for line in lines[start_line:]:
                parts = line.strip().replace(';', '').split()
                if len(parts) >= 3:
                    u = int(parts[0])
                    v = int(parts[1])
                    # SiouxFalls/Anaheim TNTP Column Order:
                    # Init, Term, Capacity, Length, FreeFlowTime, B, Power, Speed, Toll, Type
                    # Index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                    
                    capacity = float(parts[2])
                    length = float(parts[3])
                    ff_time = float(parts[4]) if len(parts) > 4 else length
                    
                    # Speed is usually index 7 (8th column)
                    # Use fallback if missing
                    if len(parts) > 7:
                        speed = float(parts[7])
                    else:
                        speed = 0.0
                        
                    # Parse is_highway (Index 10 - 11th column)
                    is_highway_val = 0.0
                    if len(parts) > 10:
                        is_highway_val = float(parts[10]) # 1.0 or 0.0
                    
                    if not self.graph.has_edge(u, v):
                        self.graph.add_edge(u, v, 
                                            capacity=capacity, 
                                            length=length, 
                                            base_weight=length, 
                                            weight=length,
                                            base_time=ff_time, 
                                            travel_time=ff_time,
                                            speed=speed,
                                            is_highway=is_highway_val, # [New]
                                            damage=0.0,
                                            status='Normal',
                                            has_building=False,
                                            total_people=0,
                                            injured=0)
        except Exception as e:
            print(f"❌ Error loading network file: {e}")
            raise e
            
        # 3. 건물 데이터 생성
        self._generate_building_data()

    def _generate_building_data(self):
        for u, v in self.graph.edges():
            if random.random() < 0.3:
                building_type = random.choice(['Small', 'Medium', 'Large'])
                if building_type == 'Small': people = random.randint(5, 15)
                elif building_type == 'Medium': people = random.randint(20, 50)
                else: people = random.randint(100, 300)
                
                self.graph[u][v].update({
                    'has_building': True,
                    'total_people': people,
                    'healthy': people,
                    'injured': 0
                })

    def apply_disaster_damage(self, damage_prob=0.3):
        for u, v in self.graph.edges():
            edge = self.graph[u][v]
            
            # 1. Damage Event Trigger
            if random.random() < damage_prob:
                # 2. Damage Severity Distribution (Skewed Control)
                # Goal: Avoid excessive 'Complete' damage (Isolation risk)
                # - Slight (40%): 0.0 ~ 0.2
                # - Moderate (30%): 0.2 ~ 0.5
                # - Extensive (25%): 0.5 ~ 0.8
                # - Complete (5%): 0.8 ~ 1.0 (Critical)
                
                severity_roll = random.random()
                if severity_roll < 0.40:
                    # Slight
                    damage = random.uniform(0.01, 0.2)
                elif severity_roll < 0.70:
                    # Moderate
                    damage = random.uniform(0.2, 0.5)
                elif severity_roll < 0.95:
                    # Extensive
                    damage = random.uniform(0.5, 0.8)
                else:
                    # Complete (Rare)
                    damage = random.uniform(0.8, 1.0)
                
                edge['damage'] = damage
                base_w = edge['base_weight']
                base_t = edge['base_time']
                
                # 3. Status Assignment (HAZUS)
                if damage > 0.8:
                    edge['status'] = 'Closed' # Complete
                    edge['weight'] = base_w * 1000.0 # Virtually infinite
                    edge['travel_time'] = base_t * 1000.0
                elif damage > 0.5:
                    edge['status'] = 'Danger' # Extensive
                    edge['weight'] = base_w * 10.0 # Expensive
                    edge['travel_time'] = base_t * 10.0
                elif damage > 0.2:
                    edge['status'] = 'Caution' # Moderate
                    edge['weight'] = base_w * 3.0
                    edge['travel_time'] = base_t * 3.0
                else:
                    edge['status'] = 'Normal' # Slight
                    edge['weight'] = base_w * 1.1 
                    edge['travel_time'] = base_t * 1.1
                    
                if edge['has_building']:
                    total_pop = edge['total_people']
                    injury_rate = damage * random.uniform(0.5, 1.0)
                    num_injured = int(total_pop * injury_rate)
                    edge['injured'] = num_injured
                    edge['healthy'] = total_pop - num_injured
            else:
                edge['damage'] = 0.0
                edge['status'] = 'Normal'
                edge['weight'] = edge['base_weight']
                edge['travel_time'] = edge['base_time']
                edge['injured'] = 0
                if edge['has_building']:
                    edge['healthy'] = edge['total_people']

    def get_shortest_path(self, start_node, end_node):
        try:
            return nx.shortest_path(self.graph, start_node, end_node, weight='weight')
        except nx.NetworkXNoPath:
            return []