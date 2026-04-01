import logging
from typing import List, Optional

# 로깅 설정
logger = logging.getLogger(__name__)

class BaseRobot:
    """
    [BaseRobot]
    Model: RoboCue-X (High-Mobility Rescue Crawler, 2026 Prototype)
    Desc: 
        도쿄 소방청 RoboCue의 'Intake Rescue' 메커니즘을 계승하되,
        2026년 시점의 '전고체 배터리(Solid-state Battery)'와 '고출력 모터'를 탑재하여
        기동성(Mobility)과 운용 시간(Operation Time)을 대폭 강화한 모델.
    """
    def __init__(self, robot_id: str, start_node: int, config: dict):
        self.id = robot_id
        self.current_node = start_node
        self.target_node = None 
        
        # --- [1] 물리 스펙 설정 (RoboCue-X 10kWh Ver.) ---
        # 무게: 200kg (안정적인 견인력 및 구조 장비 탑재)
        self.mass = config.get('mass', 200.0)
        
        # 배터리 용량: 10 kWh (현실적 타협안)
        # - Tesla Optimus(2.3kWh)의 약 4배 용량
        # - 전고체 배터리 에너지 밀도(300~400Wh/kg) 가정 시 배터리 무게 약 30kg 내외로 구현 가능
        # - 계산 편의를 위해 Joule 단위로 변환 (1 kWh = 3.6 MJ)
        self.battery_kwh_capacity = config.get('battery_kwh', 10.0)
        self.max_battery_capacity_j = self.battery_kwh_capacity * 3600000.0
        
        self.battery_j = self.max_battery_capacity_j # 현재 잔량 (Joule)
        self.battery = 100.0 # 외부 표시용 (%)
        
        # 최대 속도: 40 km/h (긴급 구조를 위한 고속 주행 설정)
        self.base_speed = config.get('speed', 40.0)
        
        # 기준 전비 (Rated Energy Efficiency): 50 Wh/km
        # - 평지(Normal)에서 200kg 로봇이 40km/h로 달릴 때의 기준 소모량
        # - 험지에서는 이 값에 '부하 계수(Load Factor)'가 곱해짐
        self.rated_efficiency_wh_per_km = 50.0 
        
        # --- 경로 및 상태 ---
        self.current_path: List[int] = []
        self.traveled_on_edge = 0.0
        self.state = "IDLE"

    def swap_battery(self):
        """
        [Battery Swap System]
        Depot이나 충전 스테이션에서 배터리를 즉시 교체하여 100%로 복구
        """
        self.battery_j = self.max_battery_capacity_j
        self.battery = 100.0
        logger.info(f"[{self.id}] Battery Swapped! Capacity: {self.battery_kwh_capacity} kWh (100%)")

    def _calculate_physics(self, length_km: float, status: str, damage: float):
        """
        [Physics Engine Core] (HAZUS 5-Level Update)
        도로 상태와 파괴도에 따른 '주행 속도' 및 '에너지 소모량' 계산
        
        Levels:
        1. None (d=0): 100% Speed, 1.0x Energy
        2. Slight (d<=0.2): 90% Speed, 1.1x Energy [Normal]
        3. Moderate (d<=0.5): 60% Speed, 1.5x Energy [Caution]
        4. Extensive (d<=0.8): 30% Speed, 3.0x Energy [Danger]
        5. Complete (d>0.8): 0% Speed, Inf Energy [Closed]
        """
        
        # 1. HAZUS Factors
        if status == 'Closed':
            # Complete Damage
            traction_factor = 0.0 # Cannot move
            load_factor = float('inf')
        elif status == 'Danger':
            # Extensive Damage
            traction_factor = 0.3
            load_factor = 3.0
        elif status == 'Caution':
            # Moderate Damage
            traction_factor = 0.6
            load_factor = 1.5
        else: # Normal
            if damage > 0.0:
                # Slight Damage
                traction_factor = 0.9
                load_factor = 1.1
            else:
                # None
                traction_factor = 1.0
                load_factor = 1.0

        # 2. Real Speed Calculation
        real_speed_kmh = self.base_speed * traction_factor
        
        # 3. Energy & Time Calculation
        if real_speed_kmh <= 1e-6:
            # Stopped / Closed
            expected_time_h = float('inf')
            energy_joule = float('inf')
            energy_percent = float('inf')
            real_speed_kmh = 0.0
        else:
            expected_time_h = length_km / real_speed_kmh
            
            # Energy = Length * (Rated * Load)
            current_efficiency_wh_km = self.rated_efficiency_wh_per_km * load_factor
            required_wh = length_km * current_efficiency_wh_km
            energy_joule = required_wh * 3600.0 # J
            energy_percent = (energy_joule / self.max_battery_capacity_j) * 100.0

        return expected_time_h, energy_percent, real_speed_kmh, energy_joule

    def predict_edge_cost(self, length: float, status: str, damage: float):
        """ 
        [External Interface] 
        GraphConverter나 Planner가 경로 비용을 예측할 때 호출
        """
        time_h, energy_pct, _, _ = self._calculate_physics(length, status, damage)
        return time_h, energy_pct

    def assign_plan(self, path: List[int]):
        """
        RL 모델이나 플래너로부터 이동 경로(노드 리스트)를 할당받음
        path 예시: [1, 2, 6, 8] (1번은 현재 위치)
        """
        if not path:
            return
            
        # 현재 위치가 경로의 시작점과 같다면, 시작점은 제외하고 다음 지점부터 저장
        if path[0] == self.current_node:
            self.current_path = path[1:]
        else:
            self.current_path = path
            
        if self.current_path:
            self.target_node = self.current_path[0]
            self.state = "MOVING"
            self.traveled_on_edge = 0.0
            logger.info(f"[{self.id}] Plan Assigned: {len(self.current_path)} steps towards Node {self.current_path[-1]}.")
        else:
            self.target_node = None
            self.state = "IDLE"

    def move(self, map_instance, dt: float = 1.0):
        """ 
        [Simulation Step]
        dt(분) 단위로 로봇을 이동시키고 상태를 업데이트
        """
        if self.state != "MOVING" or self.battery_j <= 0:
            if self.battery_j <= 0: 
                self.state = "DEPLETED"
                self.battery = 0.0 
            return

        if self.target_node is None:
            if self.current_path:
                self.target_node = self.current_path[0]
                self.traveled_on_edge = 0.0
            else:
                self.state = "IDLE"
                return

        u, v = self.current_node, self.target_node
        
        # 1. 엣지 데이터 확인
        if map_instance.graph.has_edge(u, v):
            edge_data = map_instance.graph[u][v]
            length = edge_data.get('length', 1.0)
            status = edge_data.get('status', 'Normal')
            damage = edge_data.get('damage', 0.0)
        else:
            self.state = "IDLE"
            return

        # 2. 물리 엔진 계산 (현재 상태 기준 속도/에너지 산출)
        _, _, speed_kmh, total_energy_j = self._calculate_physics(length, status, damage)

        # 3. 이동 거리 계산 (dt는 분 단위)
        step_time_h = dt / 60.0
        step_distance = speed_kmh * step_time_h
        
        # 4. 이번 스텝의 에너지 소모량 (거리에 비례하여 선형 할당)
        if length > 0:
            step_energy_j = total_energy_j * (step_distance / length)
        else:
            step_energy_j = 0

        # 5. 상태 업데이트 (배터리 차감 및 위치 이동)
        if self.battery_j >= step_energy_j:
            self.battery_j -= step_energy_j
            self.traveled_on_edge += step_distance
            
            # % 단위 업데이트 (외부 모니터링용)
            self.battery = (self.battery_j / self.max_battery_capacity_j) * 100.0
            
            # 노드 도착 처리
            if self.traveled_on_edge >= length:
                self.current_node = self.target_node
                if self.current_path: self.current_path.pop(0)
                self.traveled_on_edge = 0.0
                
                if self.current_path:
                    self.target_node = self.current_path[0]
                else:
                    self.target_node = None
                    self.state = "IDLE"
                    logger.info(f"[{self.id}] Arrived Node {self.current_node}. Bat: {self.battery:.1f}%")
        else:
            self.battery_j = 0
            self.battery = 0.0
            self.state = "DEPLETED"
            logger.warning(f"[{self.id}] Battery DEPLETED at edge {u}->{v}!")

    def get_state(self):
        return {
            "id": self.id,
            "node": self.current_node,
            "battery": round(self.battery, 2), # %
            "battery_kwh": round(self.battery_j / 3600000.0, 3), # kWh
            "state": self.state
        }

# UGV 클래스는 BaseRobot을 상속 (필요 시 특화 기능 추가 가능)
class UGV(BaseRobot):
    pass