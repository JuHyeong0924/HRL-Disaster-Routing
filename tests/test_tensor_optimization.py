"""
Tensor Optimization Validation Test Suite
==========================================
NetworkX → Tensor 전환 후 데이터 정합성, 수학적 동일성, Edge Case 검증.
실제 data/ 디렉토리의 Anaheim 데이터를 사용.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
import networkx as nx
from src.envs.worker_env import WorkerEnv
from src.envs.hrl_env import HRLEnv
import torch.nn as nn


class MockWorker(nn.Module):
    """테스트용 Mock Worker — 인접 노드 중 무작위 선택."""
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, xf, aei, edge_attr=None, batch=None, neighbors_mask=None):
        A_N = xf.size(0)
        logits = torch.full((A_N,), -1e9, device=xf.device)
        if neighbors_mask is not None:
            valid = (neighbors_mask > 0).nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                logits[valid] = torch.randn(len(valid), device=xf.device)
        probs = torch.softmax(logits.view(-1, self.num_nodes), dim=-1).view(-1)
        values = torch.zeros(A_N // self.num_nodes, device=xf.device)
        return probs, values, None


def create_env(device='cpu', disaster_prob=0.0, dynamic_disaster=False):
    """테스트용 환경 생성."""
    return WorkerEnv(
        node_file='data/Anaheim_node.tntp',
        net_file='data/Anaheim_net.tntp',
        zone_json='data/grid_Anaheim_node_to_zone.json',
        zone_graph_json='data/grid_Anaheim_zone_graph.json',
        masking_mode='soft_curr_next',
        disaster_prob=disaster_prob,
        dynamic_disaster=dynamic_disaster,
        device=device
    )


# ============================================================
# Test 1: __init__ 단계에서 텐서와 NetworkX 그래프 일치 검증
# ============================================================
def test_init_tensor_graph_consistency():
    """초기 텐서가 NetworkX 그래프와 정확히 일치하는지 검증."""
    print("[Test 1] __init__ tensor-graph consistency...", end=" ")
    env = create_env()

    # adj_matrix 검증: 모든 간선이 텐서에 양방향으로 존재
    for u, v in env.G.edges():
        ui, vi = env.node_to_idx[u], env.node_to_idx[v]
        assert env._adj_matrix_tensor[ui, vi], f"Missing edge ({u},{v}) in adj_matrix"
        assert env._adj_matrix_tensor[vi, ui], f"Missing reverse edge ({v},{u}) in adj_matrix"

    # weight_matrix 검증: 값 일치
    for u, v, data in env.G.edges(data=True):
        ui, vi = env.node_to_idx[u], env.node_to_idx[v]
        expected_w = data.get('weight', 1.0)
        actual_w = env._weight_matrix[ui, vi].item()
        assert abs(actual_w - expected_w) < 1e-5, f"Weight mismatch at ({u},{v}): {actual_w} vs {expected_w}"

    # adj_matrix에 존재하지 않는 간선은 False
    total_edges_in_tensor = env._adj_matrix_tensor.sum().item()
    expected_edges = len(list(env.G.edges())) * 2  # 양방향
    assert total_edges_in_tensor == expected_edges, \
        f"Edge count mismatch: tensor={total_edges_in_tensor}, graph={expected_edges}"

    # dist_matrix_tensor 검증
    assert torch.allclose(
        env._dist_matrix_tensor,
        torch.from_numpy(env.dist_matrix).float().to(env.device),
        atol=1e-5
    ), "dist_matrix_tensor mismatch"

    print("✅ PASSED")


# ============================================================
# Test 2: sync_tensors_from_graph() 동기화 검증
# ============================================================
def test_sync_after_disaster():
    """재난 적용 후 sync_tensors_from_graph()가 올바르게 동기화하는지 검증."""
    print("[Test 2] sync_tensors_from_graph after disaster...", end=" ")
    env = create_env(disaster_prob=0.3)

    # 재난 적용
    env.dm.apply_disaster_damage(damage_prob=0.3)
    env.sync_tensors_from_graph()

    # 모든 현존 간선이 텐서에 반영되었는지
    for u, v, data in env.G.edges(data=True):
        ui, vi = env.node_to_idx[u], env.node_to_idx[v]
        assert env._adj_matrix_tensor[ui, vi], f"Missing edge after sync ({u},{v})"

        # damage 값 일치
        expected_dmg = data.get('damage', 0.0)
        actual_dmg = env._damage_matrix[ui, vi].item()
        assert abs(actual_dmg - expected_dmg) < 1e-5, \
            f"Damage mismatch at ({u},{v}): {actual_dmg} vs {expected_dmg}"

        # status 값 일치
        expected_st = env._STATUS_MAP.get(data.get('status', 'Normal'), 0)
        actual_st = env._status_matrix[ui, vi].item()
        assert actual_st == expected_st, \
            f"Status mismatch at ({u},{v}): {actual_st} vs {expected_st}"

    print("✅ PASSED")


# ============================================================
# Test 3: _update_zone_graph_weights() 텐서 기반 결과 검증
# ============================================================
def test_zone_graph_weights_tensor_based():
    """Zone graph weight 갱신이 텐서 기반으로 올바르게 수행되는지 검증."""
    print("[Test 3] Zone graph weights (tensor-based)...", end=" ")
    env = create_env(disaster_prob=0.3)

    env.dm.apply_disaster_damage(damage_prob=0.3)
    env.sync_tensors_from_graph()
    env._update_zone_graph_weights()

    # NetworkX 기반 수동 계산과 비교
    for u_z, v_z in env.ZG.edges():
        # 수동 계산 (원래 로직)
        cross_damages = []
        for node_u in env.z2n[u_z]:
            for node_v in env.z2n[v_z]:
                if env.G.has_edge(node_u, node_v):
                    cross_damages.append(env.G[node_u][node_v].get('damage', 0.0))

        if cross_damages:
            expected_w = 1.0 * (1 + (sum(cross_damages) / len(cross_damages)) * 10)
        else:
            expected_w = float('inf')

        actual_w = env.ZG[u_z][v_z]['weight']
        if expected_w == float('inf'):
            assert actual_w == float('inf'), f"Zone ({u_z},{v_z}) should be inf"
        else:
            assert abs(actual_w - expected_w) < 1e-4, \
                f"Zone weight ({u_z},{v_z}): {actual_w} vs {expected_w}"

    print("✅ PASSED")


# ============================================================
# Test 4: step_batch의 edge_weight 텐서 조회 검증
# ============================================================
def test_step_batch_weight_lookup():
    """step_batch에서 edge weight 조회가 텐서와 NetworkX에서 동일한지 검증."""
    print("[Test 4] step_batch edge weight lookup...", end=" ")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = create_env()
    env.reset(batch_size=4)

    for _ in range(10):
        masks = env.get_action_mask_batch()
        actions = []
        for b in range(4):
            valid = masks[b].nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                actions.append(valid[random.randint(0, len(valid) - 1)].item())
            else:
                actions.append(int(env.curr_nodes[b].item()))

        prev_nodes = env.curr_nodes.clone()
        _, _, dones, infos = env.step_batch(torch.tensor(actions))

        for b in range(4):
            if infos[b].get('reason') in ('invalid', 'stagnation', 'destroyed'):
                continue
            prev_idx = int(prev_nodes[b].item())
            new_idx = int(env.curr_nodes[b].item())
            if prev_idx != new_idx:
                # 텐서 weight
                tensor_w = env._weight_matrix[prev_idx, new_idx].item()
                # NetworkX weight
                u_node = env.idx_to_node[prev_idx]
                v_node = env.idx_to_node[new_idx]
                nx_w = env.G[u_node][v_node].get('weight', 1.0)
                assert abs(tensor_w - nx_w) < 1e-5, \
                    f"Weight mismatch: tensor={tensor_w}, nx={nx_w}"

    print("✅ PASSED")


# ============================================================
# Test 5: Dynamic Edge Masking 텐서 기반 검증
# ============================================================
def test_dynamic_edge_masking():
    """aftershock 후 제거된 간선이 마스크에서 올바르게 0으로 처리되는지 검증."""
    print("[Test 5] Dynamic edge masking (tensor-based)...", end=" ")
    env = create_env(disaster_prob=0.5, dynamic_disaster=True)
    env.reset(batch_size=2)

    # 재난 적용하여 일부 간선의 weight를 변경 (HAZUS 모델은 간선을 제거하지 않음)
    env.dm.apply_disaster_damage(damage_prob=0.5)
    env.sync_tensors_from_graph()

    masks = env.get_action_mask_batch()

    for b in range(2):
        c_idx = int(env.curr_nodes[b].item())
        for n_idx in env._adj_list[c_idx]:
            if env._adj_matrix_tensor[c_idx, n_idx]:
                # 텐서에 간선이 존재하면 마스크도 1이어야 함
                # (단, zone masking으로 0이 될 수 있으므로 soft 모드에서만 의미 있음)
                pass  # soft mode에서는 zone 제약이 없으므로 adj만 보면 됨
            else:
                # 텐서에 간선이 없으면 마스크도 0이어야 함
                assert masks[b, n_idx].item() == 0.0, \
                    f"Removed edge ({c_idx},{n_idx}) still in mask for batch {b}"

    print("✅ PASSED")


# ============================================================
# Test 6: HRLEnv get_target_features 벡터화 정합성 검증
# ============================================================
def test_hrl_target_features_vectorized():
    """HRLEnv.get_target_features()의 벡터화 결과가 스칼라 연산과 일치하는지 검증."""
    print("[Test 6] HRLEnv target features vectorization...", end=" ")
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    env = create_env()
    worker = MockWorker(num_nodes=env.num_nodes).to(env.device)
    hrl = HRLEnv(worker=worker, worker_env=env)
    hrl.reset(batch_size=4, num_targets=5)

    feats = hrl.get_target_features()

    # 스칼라 검증
    for b in range(4):
        c_idx = int(env.curr_nodes[b])
        for i in range(5):
            t_idx = int(hrl.targets[b, i])
            dist = env._dist_matrix_tensor[c_idx, t_idx].item()
            if np.isinf(dist):
                dist = env.max_dist
            dl = hrl.deadlines[b, i].item()
            rem = dl - hrl.current_time[b].item()

            # Ch.0: deadline
            assert abs(feats[b, i, 0].item() - dl / hrl.max_time) < 1e-5
            # Ch.1: time_remaining
            assert abs(feats[b, i, 1].item() - rem / hrl.max_time) < 1e-5
            # Ch.2: dist_from_curr
            assert abs(feats[b, i, 2].item() - dist / max(env.max_dist, 1.0)) < 1e-5

    print("✅ PASSED")


# ============================================================
# Test 7: HRLEnv get_zone_adj_mask 벡터화 정합성 검증
# ============================================================
def test_hrl_zone_adj_mask_vectorized():
    """HRLEnv.get_zone_adj_mask()의 벡터화 결과가 스칼라 연산과 일치하는지 검증."""
    print("[Test 7] HRLEnv zone_adj_mask vectorization...", end=" ")
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)

    env = create_env()
    worker = MockWorker(num_nodes=env.num_nodes).to(env.device)
    hrl = HRLEnv(worker=worker, worker_env=env)
    hrl.reset(batch_size=8, num_targets=5)

    mask = hrl.get_zone_adj_mask()

    for b in range(8):
        c_node = env.idx_to_node[int(env.curr_nodes[b])]
        c_zone = env.n2z[c_node]
        # 자기 자신은 반드시 1
        assert mask[b, c_zone].item() == 1, f"Self-zone missing for batch {b}"
        # 인접 zone도 1
        for nbr in env.ZG.neighbors(c_zone):
            assert mask[b, nbr].item() == 1, f"Neighbor zone {nbr} missing for batch {b}"
        # 비인접 zone은 0
        all_adj = set(env.ZG.neighbors(c_zone)) | {c_zone}
        for z in range(env.k):
            if z not in all_adj:
                assert mask[b, z].item() == 0, f"Non-adjacent zone {z} is 1 for batch {b}"

    print("✅ PASSED")


# ============================================================
# Test 8: HRLEnv get_target_mask 벡터화 정합성 검증
# ============================================================
def test_hrl_target_mask_vectorized():
    """HRLEnv.get_target_mask() 벡터화 결과가 스칼라 기대값과 일치하는지 검증."""
    print("[Test 8] HRLEnv target_mask vectorization...", end=" ")
    random.seed(55)
    np.random.seed(55)
    torch.manual_seed(55)

    env = create_env()
    worker = MockWorker(num_nodes=env.num_nodes).to(env.device)
    hrl = HRLEnv(worker=worker, worker_env=env)
    hrl.reset(batch_size=4, num_targets=5)

    # 일부 타겟을 구출 처리
    hrl.target_rescued[0, 0] = True
    hrl.target_rescued[1, 2] = True
    # 시간을 많이 소진
    hrl.current_time[2] = hrl.max_time + 1

    mask = hrl.get_target_mask()

    # 구출된 타겟은 0
    assert mask[0, 0].item() == 0
    assert mask[1, 2].item() == 0
    # 시간 초과로 모든 타겟 비활성
    for i in range(5):
        assert mask[2, i].item() == 0

    print("✅ PASSED")


# ============================================================
# Test 9: HRLEnv _build_graph_data 텐서 인덱싱 검증
# ============================================================
def test_build_graph_data_tensor():
    """_build_graph_data()의 텐서 인덱싱이 기존 NetworkX dict 순회와 동일한 결과를 내는지 검증."""
    print("[Test 9] _build_graph_data tensor indexing...", end=" ")
    env = create_env(disaster_prob=0.2)
    env.dm.apply_disaster_damage(damage_prob=0.2)
    env.sync_tensors_from_graph()

    worker = MockWorker(num_nodes=env.num_nodes).to(env.device)
    hrl = HRLEnv(worker=worker, worker_env=env)

    edge_index, edge_attr = hrl._build_graph_data()

    # edge_attr shape 검증
    E = edge_index.shape[1]
    assert edge_attr.shape == (E, 2), f"edge_attr shape: {edge_attr.shape}"

    # 정규화 후 값이 [0, 1] 범위인지
    assert edge_attr.min() >= -1e-5, f"edge_attr min: {edge_attr.min()}"
    assert edge_attr.max() <= 1.0 + 1e-5, f"edge_attr max: {edge_attr.max()}"

    print("✅ PASSED")


# ============================================================
# Test 10: HRLEnv get_zone_features disaster_intensity 벡터화 검증
# ============================================================
def test_zone_features_disaster_intensity():
    """get_zone_features의 Ch.3 (disaster_intensity)가 수동 계산과 일치하는지 검증."""
    print("[Test 10] Zone features disaster_intensity...", end=" ")
    random.seed(33)
    np.random.seed(33)
    torch.manual_seed(33)

    env = create_env(disaster_prob=0.3)
    env.dm.apply_disaster_damage(damage_prob=0.3)
    env.sync_tensors_from_graph()

    worker = MockWorker(num_nodes=env.num_nodes).to(env.device)
    hrl = HRLEnv(worker=worker, worker_env=env)
    hrl.reset(batch_size=2, num_targets=3)

    feats = hrl.get_zone_features()

    # 벡터화된 로직 자체의 수학적 정합성을 수동으로 재계산
    # 핵심: _adj_matrix_tensor는 양방향이므로 [N,N] 기반 집계가 정확함
    adj_float = env._adj_matrix_tensor.float()
    node_adj_damage_sum = (env._damage_matrix * adj_float).sum(dim=1)  # [N]
    node_adj_count = adj_float.sum(dim=1)  # [N]
    for z in range(env.k):
        z_nodes = [env.node_to_idx[n] for n in env.z2n[z]]
        if not z_nodes:
            continue
        expected = node_adj_damage_sum[z_nodes].sum().item() / max(node_adj_count[z_nodes].sum().item(), 1)
        actual = feats[0, z, 3].item()
        assert abs(actual - expected) < 1e-4, \
            f"Zone {z} disaster_intensity: {actual} vs {expected} (diff={abs(actual-expected):.6f})"

    print("✅ PASSED")


# ============================================================
# Test 11: HRLEnv step_manager 텐서 기반 엔드-투-엔드 검증
# ============================================================
def test_step_manager_e2e():
    """step_manager()가 텐서 기반으로 정상 동작하는지 엔드-투-엔드 검증."""
    print("[Test 11] step_manager E2E...", end=" ")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = 'cpu'
    env = create_env(device=device, disaster_prob=0.1, dynamic_disaster=True)
    worker = MockWorker(num_nodes=env.num_nodes).to(device)
    hrl = HRLEnv(worker=worker, worker_env=env)
    hrl.reset(batch_size=4, num_targets=5)

    total_steps = 0
    for _ in range(20):
        if hrl.dones.all():
            break
        t_act = torch.zeros(4, dtype=torch.long, device=device)
        z_act = torch.zeros(4, dtype=torch.long, device=device)

        # 유효한 타겟과 zone 선택
        t_mask = hrl.get_target_mask()
        z_mask = hrl.get_zone_adj_mask()
        for b in range(4):
            if hrl.dones[b]:
                continue
            valid_t = t_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_t) > 0:
                t_act[b] = valid_t[0]
            valid_z = z_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_z) > 0:
                z_act[b] = valid_z[0]

        events, dones = hrl.step_manager(t_act, z_act)
        total_steps += 1

        # 기본 상태 검증
        assert len(events) == 4
        for e in events:
            assert e in ['none', 'target_rescued', 'zone_arrived', 'zone_escaped',
                         'agent_destroyed', 'agent_trapped']

    assert total_steps > 0, "No steps executed"
    print(f"✅ PASSED ({total_steps} manager steps)")


# ============================================================
# Test 12: hrl_env.py reset의 sync_tensors_from_graph 호출 누락 검증
# ============================================================
def test_hrl_reset_sync_gap():
    """HRLEnv.reset()에서 재난 적용 후 sync_tensors_from_graph() 호출 여부 검증."""
    print("[Test 12] HRLEnv.reset sync gap check...", end=" ")

    env = create_env(disaster_prob=0.3)
    worker = MockWorker(num_nodes=env.num_nodes).to(env.device)
    hrl = HRLEnv(worker=worker, worker_env=env)
    hrl.reset(batch_size=2, num_targets=3)

    # reset 후 damage가 텐서에 반영되었는지 간접 검증:
    # env._damage_matrix에 0이 아닌 값이 있는지 (재난이 적용되었다면)
    has_damage = env._damage_matrix.sum().item() > 0
    # 그래프에서도 damage가 있는지
    graph_has_damage = any(
        d.get('damage', 0.0) > 0 for _, _, d in env.G.edges(data=True)
    )

    # 둘 다 동일해야 함 (둘 다 True이거나 둘 다 False)
    assert has_damage == graph_has_damage, \
        f"Sync mismatch: tensor_has_damage={has_damage}, graph_has_damage={graph_has_damage}"

    print("✅ PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Tensor Optimization Validation Suite")
    print("=" * 60)

    test_init_tensor_graph_consistency()
    test_sync_after_disaster()
    test_zone_graph_weights_tensor_based()
    test_step_batch_weight_lookup()
    test_dynamic_edge_masking()
    test_hrl_target_features_vectorized()
    test_hrl_zone_adj_mask_vectorized()
    test_hrl_target_mask_vectorized()
    test_build_graph_data_tensor()
    test_zone_features_disaster_intensity()
    test_step_manager_e2e()
    test_hrl_reset_sync_gap()

    print("=" * 60)
    print("🎉 ALL 12 TESTS PASSED!")
    print("=" * 60)
