import torch

t1 = torch.tensor([1.0, 2.0], requires_grad=True)
full = torch.zeros(3)
active = [0, 2]
full[active] = t1
loss = full.sum()
loss.backward()
print("Grad:", t1.grad)
