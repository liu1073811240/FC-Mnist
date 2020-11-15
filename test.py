import torch

a = torch.randn([10, 2], requires_grad=True)
print(a)
print(a.detach().numpy())  # 转成numpy数据并且只取数据。适用于矩阵，向量的情况

b = torch.randn([1], requires_grad=True)
print(b.item())  # 只能取标量



