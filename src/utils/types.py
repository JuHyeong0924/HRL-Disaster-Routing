from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

@dataclass
class Task:
    """
    임무 객체 (Task)
    설명: 환경(지도) 상의 특정 노드에서 생성되어 로봇에게 할당되는 작업 단위입니다.
    """
    task_id: int                    # 고유 ID
    task_type: str                  # "RECON"(정찰), "RESCUE"(구조), "SUPPLY"(보급)
    node_id: int                    # 지도 그래프 상의 노드 인덱스
    location: Tuple[float, float]   # (x, y) 절대 좌표
    priority: int                   # 1(하) ~ 3(상)
    status: str                     # "PENDING", "ASSIGNED", "COMPLETED"
    
    # 할당된 에이전트 ID (없으면 None)
    assigned_agent_id: Optional[str] = None 

    # 임무 수행에 필요한 자원 및 제약 조건 (단위: 분, %)
    # 예시: {"work_time": 30.0, "battery_cost": 5.0}
    required_resources: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    로봇 상태 객체 (AgentState)
    설명: 시뮬레이터 및 학습 모델이 참조하는 로봇의 실시간 상태 정보입니다.
    """
    agent_id: str                   # 로봇 ID
    agent_type: str                 # "UAV", "UGV"

    # --- 위치 정보 ---
    current_node: int               # 현재 머무는 노드 ID (이동 중이거나 노드 위가 아니면 -1)
    current_edge: Tuple[int, int]   # 이동 중인 엣지 (출발노드, 도착노드), 없으면 빈 튜플 또는 (-1, -1) 처리 필요
    position: Tuple[float, float]   # (x, y) 절대 좌표

    # --- 상태 정보 ---
    battery: float                  # 잔여 배터리 (0.0 ~ 100.0)
    status: str                     # "IDLE", "MOVING", "WORKING", "RESUPPLYING"
                                    # WORKING: 구조/정찰 수행 중, RESUPPLYING: 배터리 교체 중

    # --- 임무 및 경로 ---
    assigned_task_queue: List[int]  # 수행해야 할 임무 ID 리스트
    current_path: List[int]         # 현재 목표로 가기 위한 경로 노드 리스트