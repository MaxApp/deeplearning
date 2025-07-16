"""
二次函数曲线拟合
模型：y = 3*x^2 + 20
"""

import matplotlib.pyplot as plt
import torch

# y = 3(*x**2) + 20
def plot_function(x, y, y_pred, title='Function Plot', xlabel='x', ylabel='y'):
    plt.figure(figsize=(10, 6))
    plt.scatter(x.detach().numpy(), y.detach().numpy(), color='blue', label='Data Points')
    plt.plot(x.detach().numpy(), y_pred.detach().numpy(), label='y_pred', color='red')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.pause(1.5)
    plt.close()

# def scatter_plot(x, y, title='Scatter Plot', xlabel='x', ylabel='y'):
#     plt.figure(figsize=(10, 6))
    
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# 生成数据
x = torch.linspace(-10, 10, 50).view(-1, 1)  # 从-10到10生成50个点
y = 3 * (x ** 2) + 20 + (5*torch.randn(1,50)).view(-1, 1) # 计算对应y值

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        torch.manual_seed(42)  # 设置随机种子以确保可重复性
        self.linear1 = torch.nn.Linear(1, 10)  # 输入维度为1，输出维度为1
        self.linear2 = torch.nn.Linear(10, 20)  # 输入维度为10，输出维度为1
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x)) 
        return self.linear(x)

# 创建模型实例
model = MyNetwork()
# 损失函数
loss_func = torch.nn.MSELoss()
# 优化器
torch_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 训练模型
epochs = 1500
for epoch in range(epochs):
    y_pred = model(x)  # 前向传播
    loss = loss_func(y_pred, y)  # 计算损失
    torch_optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 反向传播
    torch_optimizer.step()  # 更新参数
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        # 绘制函数图像
        plot_function(x, y, y_pred)

