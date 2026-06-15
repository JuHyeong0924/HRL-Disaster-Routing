# State 02: Blueprint

## Architectural Optimization: Vectorized Environment

### Identified Bottlenecks (CPU/GPU Starvation)
1. **Inefficient Tensor Assembly (`torch.stack` overhead):** 
   In `hrl_env.py`, the state and mask tensors are gathered via list comprehensions: `torch.stack([st[b] for b in active])`. Since `st` is already a batched tensor of shape `[B, N, D]`, this slicing and restacking forces PyTorch to allocate `len(active)` separate memory blocks and concatenate them on the CPU, causing massive `1368%` CPU load.
2. **Unvectorized Python `for` loops in Env:** 
   `WorkerEnv._get_state_batch()` and `WorkerEnv.get_action_mask_batch()` iterate over `B` items sequentially in Python. 
3. **Dictionary Lookups in Hot Loop:**
   `self._adj_list` is a Python list of lists, causing branch mispredictions and slow CPU memory lookups during the adjacency checks.

### Proposed Changes

#### 1. `src/envs/hrl_env.py` (Zero-Copy Slicing)
- **[MODIFY] `step_manager()`:**
  - Replace `xs = torch.stack([st[b].to(device) for b in active])` with `xs = st[active].to(device)`
  - Replace `ms = torch.stack([self.env.get_action_mask_batch()[b].to(device) for b in active])` with `ms = self.env.get_action_mask_batch()[active].to(device)`

#### 2. `src/envs/worker_env.py` (Vectorized Tensor Operations)
- **[MODIFY] Initialization:** 
  - Convert `self._adj_list` into a dense boolean adjacency matrix tensor `self._adj_matrix_tensor` of shape `[N, N]` on the CPU.
- **[MODIFY] `_get_state_batch()`:**
  - Eliminate the `for b in range(B)` loop.
  - Use PyTorch Advanced Indexing: `state[torch.arange(B), self.curr_nodes, 0] = 1.0`
- **[MODIFY] `get_action_mask_batch()`:**
  - Eliminate the `for b in range(B)` loop.
  - Use `mask = self._adj_matrix_tensor[self.curr_nodes]` to fetch valid neighbors for all batches simultaneously in C++ backend.
