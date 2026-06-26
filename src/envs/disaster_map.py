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
                    # 시속 30km/h (약 1640.42 ft/min) 기준으로 분(minute) 단위 소요 시간 산출
                    speed_ft_per_min = 1640.42
                    travel_time_min = length / speed_ft_per_min
                    
                    ff_time = float(parts[4]) if len(parts) > 4 else travel_time_min
                    
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
                                            base_weight=travel_time_min, 
                                            weight=travel_time_min,
                                            base_time=travel_time_min, 
                                            travel_time=travel_time_min,
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
        
        # 4. 원본 간선 백업 (리셋 시 복원용, 빌딩 데이터 포함)
        self._original_edges = {(u, v): dict(d) for u, v, d in self.graph.edges(data=True)}

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
        is_reset = (damage_prob <= 0.0)
        
        # ── 리셋 모드: 제거된 간선 복원 후 어트리뷰트 초기화 ──
        if is_reset:
            # 1단계: 제거된 간선 복원
            for (u, v), attrs in self._original_edges.items():
                if not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v, **attrs)
            # 2단계: 모든 간선 어트리뷰트 초기화
            for u, v in self.graph.edges():
                edge = self.graph[u][v]
                edge['damage'] = 0.0
                edge['status'] = 'Normal'
                edge['weight'] = edge['base_weight']
                edge['travel_time'] = edge['base_time']
                edge['injured'] = 0
                if edge['has_building']:
                    edge['healthy'] = edge['total_people']
            return
        
        # ── 데미지 모드: 누적 데미지 + HAZUS Residual Capacity 기반 Soft Closure ──
        # [HAZUS Earthquake Model Technical Manual, Transportation Systems Chapter]
        # Weight Multiplier = 1 / Residual Capacity
        # Complete(>0.8) 간선은 제거하지 않고 가중치를 극대화 (Soft Closure)
        # → UGV(무인 지상 차량)는 특수 구조 장비로 강행 돌파 가능하나 파괴 위험 극심
        for u, v in list(self.graph.edges()):  # list()로 복사: 반복 중 변경 방지
            edge = self.graph[u][v]
            
            if random.random() < damage_prob:
                severity_roll = random.random()
                if severity_roll < 0.40:
                    new_damage = random.uniform(0.01, 0.2)
                elif severity_roll < 0.70:
                    new_damage = random.uniform(0.2, 0.5)
                elif severity_roll < 0.95:
                    new_damage = random.uniform(0.5, 0.8)
                else:
                    new_damage = random.uniform(0.8, 1.0)
                
                # 기존 데미지와 누적 합산 (최대 1.0)
                damage = min(1.0, edge.get('damage', 0.0) + new_damage)
                edge['damage'] = damage
                base_w = edge['base_weight']
                base_t = edge['base_time']
                
                # HAZUS Residual Capacity 기반 등급 & 가중치 할당
                if damage > 0.8:
                    # Complete (Closed): Residual Capacity ~5% (UGV 강행 돌파)
                    # UGV 통과 시 30% 파괴 확률 (hrl_env.py에서 판정)
                    edge['status'] = 'Closed'
                    edge['weight'] = base_w * 20.0
                    edge['travel_time'] = base_t * 20.0
                elif damage > 0.5:
                    # Extensive (Danger): Residual Capacity 25%
                    edge['status'] = 'Danger'
                    edge['weight'] = base_w * 4.0
                    edge['travel_time'] = base_t * 4.0
                elif damage > 0.2:
                    # Moderate (Caution): Residual Capacity 50%
                    edge['status'] = 'Caution'
                    edge['weight'] = base_w * 2.0
                    edge['travel_time'] = base_t * 2.0
                else:
                    # Slight (Normal): Residual Capacity 100%
                    edge['status'] = 'Normal'
                    edge['weight'] = base_w * 1.0
                    edge['travel_time'] = base_t * 1.0
                    
                if edge['has_building']:
                    total_pop = edge['total_people']
                    injury_rate = damage * random.uniform(0.5, 1.0)
                    num_injured = int(total_pop * injury_rate)
                    edge['injured'] = num_injured
                    edge['healthy'] = total_pop - num_injured

    def get_shortest_path(self, start_node, end_node):
        try:
            return nx.shortest_path(self.graph, start_node, end_node, weight='weight')
        except nx.NetworkXNoPath:
            return []