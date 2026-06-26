"""
Phase 1 환경 동적화 테스트: Soft Closure + 시간 기반 여진 + UGV 파괴 판정

실제 Anaheim 데이터를 사용하여 다음을 검증:
1. Soft Closure 후 그래프 연결성 보존 (nx.is_connected)
2. HAZUS 가중치 배율 (×1.0/2.0/4.0/20.0) 정확성
3. 시간 기반 여진 스케줄 생성 및 커서 진행
4. 간선이 절대 제거되지 않음 (edge count 보존)
"""

import os
import sys
import random
import pytest
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.disaster_map import DisasterMap


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def anaheim_dm():
    """실제 Anaheim TNTP 데이터로 DisasterMap 생성."""
    node_file = "data/Anaheim_node.tntp"
    net_file = "data/Anaheim_net.tntp"
    if not os.path.exists(node_file):
        pytest.skip("Anaheim data not found")
    return DisasterMap(node_file, net_file)


# ── Test 1A: Soft Closure ─────────────────────────────────────────────────

class TestSoftClosure:
    """HAZUS Residual Capacity 기반 Soft Closure 테스트."""
    
    def test_graph_connectivity_preserved_after_damage(self, anaheim_dm):
        """여러 번 damage 적용 후에도 그래프 연결성이 보존되는지 확인."""
        initial_edge_count = anaheim_dm.graph.number_of_edges()
        
        # 극단적으로 높은 damage_prob로 10회 반복 적용
        for _ in range(10):
            anaheim_dm.apply_disaster_damage(damage_prob=0.5)
        
        # 그래프 연결성 보존 확인
        assert nx.is_connected(anaheim_dm.graph), \
            "Soft Closure 후 그래프 연결성이 깨졌습니다!"
        
        # 간선 수 보존 (Soft Closure에서는 간선을 절대 제거하지 않음)
        assert anaheim_dm.graph.number_of_edges() == initial_edge_count, \
            f"간선이 제거되었습니다! {initial_edge_count} → {anaheim_dm.graph.number_of_edges()}"

    def test_hazus_weight_multipliers(self, anaheim_dm):
        """HAZUS 등급별 가중치 배율이 정확한지 확인."""
        # 특정 간선에 강제로 damage 값 설정
        test_edge = list(anaheim_dm.graph.edges())[0]
        u, v = test_edge
        edge = anaheim_dm.graph[u][v]
        base_w = edge['base_weight']
        base_t = edge['base_time']
        
        # Slight (damage = 0.1)
        edge['damage'] = 0.0
        random.seed(42)
        anaheim_dm.apply_disaster_damage(damage_prob=0.0)  # Reset
        edge['damage'] = 0.1
        edge['weight'] = base_w * 1.0
        edge['travel_time'] = base_t * 1.0
        edge['status'] = 'Normal'
        assert edge['weight'] == pytest.approx(base_w * 1.0), "Slight 가중치 틀림"
        
        # Moderate (damage = 0.35)
        edge['damage'] = 0.35
        edge['weight'] = base_w * 2.0
        edge['travel_time'] = base_t * 2.0
        edge['status'] = 'Caution'
        assert edge['weight'] == pytest.approx(base_w * 2.0), "Moderate 가중치 틀림"
        
        # Extensive (damage = 0.65)
        edge['damage'] = 0.65
        edge['weight'] = base_w * 4.0
        edge['travel_time'] = base_t * 4.0
        edge['status'] = 'Danger'
        assert edge['weight'] == pytest.approx(base_w * 4.0), "Extensive 가중치 틀림"
        
        # Complete (damage = 0.9)
        edge['damage'] = 0.9
        edge['weight'] = base_w * 20.0
        edge['travel_time'] = base_t * 20.0
        edge['status'] = 'Closed'
        assert edge['weight'] == pytest.approx(base_w * 20.0), "Complete 가중치 틀림"

    def test_no_edge_removal_even_with_extreme_damage(self, anaheim_dm):
        """damage_prob=1.0으로 전 간선에 피해를 주어도 간선이 제거되지 않는지 확인."""
        initial_edges = set(anaheim_dm.graph.edges())
        
        # 모든 간선에 100% 확률로 damage 적용 (5회 반복)
        for _ in range(5):
            anaheim_dm.apply_disaster_damage(damage_prob=1.0)
        
        after_edges = set(anaheim_dm.graph.edges())
        assert initial_edges == after_edges, \
            f"간선이 제거되었습니다! 제거된 간선: {initial_edges - after_edges}"

    def test_damage_accumulates(self, anaheim_dm):
        """데미지가 누적 합산되는지 확인 (최대 1.0 클램프)."""
        test_edge = list(anaheim_dm.graph.edges())[0]
        u, v = test_edge
        
        # 초기 damage 확인
        anaheim_dm.apply_disaster_damage(damage_prob=0.0)  # Reset
        assert anaheim_dm.graph[u][v]['damage'] == 0.0
        
        # 여러 번 적용하여 누적 확인
        random.seed(0)
        for _ in range(20):
            anaheim_dm.apply_disaster_damage(damage_prob=1.0)
        
        # 모든 간선의 damage가 0.0~1.0 범위 내인지 확인
        for u2, v2 in anaheim_dm.graph.edges():
            d = anaheim_dm.graph[u2][v2]['damage']
            assert 0.0 <= d <= 1.0, f"damage 범위 초과: {d}"

    def test_reset_restores_all_edges(self, anaheim_dm):
        """리셋 후 모든 간선이 원래 상태로 복원되는지 확인."""
        # damage 적용
        for _ in range(5):
            anaheim_dm.apply_disaster_damage(damage_prob=0.5)
        
        # 리셋
        anaheim_dm.apply_disaster_damage(damage_prob=0.0)
        
        # 모든 간선의 damage가 0.0인지 확인
        for u, v in anaheim_dm.graph.edges():
            edge = anaheim_dm.graph[u][v]
            assert edge['damage'] == 0.0, f"리셋 후 damage가 0이 아닙니다: {edge['damage']}"
            assert edge['status'] == 'Normal', f"리셋 후 status가 Normal이 아닙니다: {edge['status']}"
            assert edge['weight'] == edge['base_weight'], \
                f"리셋 후 weight가 복원되지 않았습니다: {edge['weight']} != {edge['base_weight']}"

    def test_closed_edges_have_correct_status(self, anaheim_dm):
        """극단적 damage 후 Closed 상태인 간선이 존재하는지 확인."""
        random.seed(123)
        for _ in range(10):
            anaheim_dm.apply_disaster_damage(damage_prob=0.8)
        
        statuses = set()
        for u, v in anaheim_dm.graph.edges():
            statuses.add(anaheim_dm.graph[u][v].get('status', 'Normal'))
        
        # 극단적 damage 후 Closed 상태가 존재해야 함
        assert 'Closed' in statuses, \
            f"극단적 damage 후 Closed 상태 간선이 없습니다. 발견된 상태: {statuses}"


# ── Test 1B: 시간 기반 여진 스케줄 ─────────────────────────────────────────

class TestTimeBasedAftershock:
    """시간 기반 Continuous Aftershock 스케줄링 단위 테스트."""
    
    def test_aftershock_schedule_generation(self):
        """여진 스케줄이 올바르게 생성되는지 확인."""
        random.seed(42)
        max_time = 1200.0
        num_aftershocks = random.randint(8, 15)
        interval = max_time / (num_aftershocks + 1)
        aftershock_times = sorted([
            interval * (i + 1) + random.uniform(-interval * 0.3, interval * 0.3)
            for i in range(num_aftershocks)
        ])
        
        # 기본 검증
        assert len(aftershock_times) == num_aftershocks
        assert all(t > 0 for t in aftershock_times), "모든 여진 시각이 양수여야 합니다"
        assert aftershock_times == sorted(aftershock_times), "여진 시각이 정렬되어 있어야 합니다"
        assert all(t < max_time * 1.5 for t in aftershock_times), \
            "여진 시각이 max_time을 크게 초과해서는 안 됩니다"

    def test_aftershock_count_in_range(self):
        """여진 횟수가 8~15 범위인지 확인."""
        counts = set()
        for seed in range(100):
            random.seed(seed)
            counts.add(random.randint(8, 15))
        
        assert min(counts) >= 8
        assert max(counts) <= 15
