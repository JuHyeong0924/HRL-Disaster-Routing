import torch
import evaluate
from evaluate import visualize_map, MAP_CONFIGS
from src.models.manager import Manager
from src.models.worker import Worker
from train_rl import _load_worker_checkpoint, _load_manager_checkpoint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

worker = Worker(node_dim=4, hidden_dim=256, num_layers=2).to(device)
worker_ckpt = 'logs/rl_worker_stage/2026-06-09_2351_worker_B32/best.pt'
_load_worker_checkpoint(worker_ckpt, worker, device, [])
worker.eval()

manager = Manager(node_dim=4, hidden_dim=256, num_layers=2).to(device)
manager_ckpt = 'logs/rl_manager_stage/2026-06-10_0029_manager_B32/best.pt'
_load_manager_checkpoint(manager_ckpt, manager, device, [])
manager.eval()

print("Visualizing Berlin Mitte...")
visualize_map('berlin-mitte', worker, manager, device)
print("Visualizing Berlin Friedrichshain...")
visualize_map('berlin-friedrichshain', worker, manager, device)
print("Done!")
