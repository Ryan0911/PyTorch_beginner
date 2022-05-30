import torch
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does 'a' require gradients? : {a.requires_grad}")
b = x + z
print(f"Does 'b' require gradients? : {b.requires_grad}")
