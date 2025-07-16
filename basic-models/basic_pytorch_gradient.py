"""
basic_manual_gradient.py 的 PyTorch 版本
"""

import torch

# y = w1*x1+w2*x2+b
# y = 2*x1 + 3*x2 + 5

x = torch.tensor([[0, 1, 0, 1, 2, 0, 2, 1, 3, 0, 3, 1, 4, 0, 2],[0, 0, 1, 1, 0, 2, 1, 2, 0, 3, 1, 3, 0, 4, 2]], dtype=torch.float32)
y = torch.tensor([5, 8, 7, 10, 9, 11, 12, 13, 11, 14, 14, 16, 13, 17, 15], dtype=torch.float32)
# random.seed(42)
# 随机生成 w
# w = torch.randn((1,2), requires_grad=True)
# b = torch.randn(1, requires_grad=True)
# 打印初始权重
# print(f"Initial weights: w1={w[0][0].item()}, w2={w[0][1].item()}, b={b.item()}")
# 学习率
lr = 0.001

# 模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # 输入维度为2，输出维度为1

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# 损失函数
loss_func = torch.nn.MSELoss()
# 参数优化器
torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 迭代次数
epochs = 1000
# 训练模型
for epoch in range(epochs):
    y_pred = model(x)  # 前向传播
    loss = loss_func(y_pred.squeeze(), y)  # 计算损失
    torch_optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 反向传播
    torch_optimizer.step()  # 更新参数
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
# 打印最终权重
print(f"Final weights: w1={model.linear.weight[0][0].item()}, w2={model.linear.weight[0][1].item()}, b={model.linear.bias.item()}")



    