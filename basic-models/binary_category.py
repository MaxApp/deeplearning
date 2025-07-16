""" 二分类回归 """

import torch
import matplotlib.pyplot as plt

sample_num = 60
x0_mean_value = 4.0
x1_mean_value = 7.0
y0_mean_value = 5.0
y1_mean_value = 8.0
standard = 1.2
m_data = torch.ones(sample_num, 1)  # 生成一个形状为 (sample_num, 1) 的全1张量
x0 = torch.normal(x0_mean_value * m_data, std=standard)
x1 = torch.normal(x1_mean_value * m_data, std=standard)
y0 = torch.normal(y0_mean_value * m_data, std=standard)
y1 = torch.normal(y1_mean_value * m_data, std=standard)

tag0 = torch.zeros(sample_num, 1)  # 类别0的标签
tag1 = torch.ones(sample_num, 1)   # 类别1的标签
train_x = torch.cat((x0, x1), dim=0)
train_y = torch.cat((y0, y1), dim=0)
tag_lable = torch.cat((tag0, tag1), dim=0)

# plt.scatter(x0.numpy(), y0.numpy(), color='blue', label='Category 0')
# plt.scatter(x1.numpy(), y1.numpy(), color='red', label='Category 1')
# plt.title('Binary Category Scatter Plot')
# plt.xlabel('x')
# plt.ylabel('y') 
# plt.show()

class BinaryCategory(torch.nn.Module):
    def __init__(self):
        super(BinaryCategory, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
        # self.linear1 = torch.nn.Linear(2, 5)  # 输入维度为2，输出维度为10
        # self.linear2 = torch.nn.Linear(5, 8)
        # self.linear3 = torch.nn.Linear(8, 1)

    def forward(self, xy):
        return torch.sigmoid(self.linear(xy))
        # x = torch.sigmoid(self.linear1(x))
        # x = torch.sigmoid(self.linear2(x))
        # return torch.sigmoid(self.linear3(x))  # 使用sigmoid激活函数进行二分类
    
loss_func = torch.nn.BCELoss()  # 二分类交叉熵损失函数
model = BinaryCategory()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 使用SGD优化器

epochs = 1800
plt.ion()  # 开启交互模式
plt.xlim(-2, 10)
plt.ylim(-2, 20)
for epoch in range(epochs):
    # 随机从 train_x 和 train_y 中选择40个样本进行训练
    indices = torch.randperm(train_x.size(0))[:40]
    x_batch = train_x[indices].view(-1, 1)  # 将 x_batch 形状调整为 (40, 1)
    y_batch = train_y[indices].view(-1, 1)  # 将 y_batch 形状调整为 (40, 1)
    tag_batch = tag_lable[indices].view(-1, 1)  # 将 tag_batch 形状调整为 (40, 1)
    inputs = torch.cat((x_batch, y_batch), dim=1)  # 合并 x_batch 和 y_batch
    outputs = model(inputs)  # 前向传播
    loss = loss_func(outputs, tag_batch)  # 计算损失
    optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    if (epoch + 1) % 100 == 0:
        mask = outputs > (0.5)
        correct = (mask == tag_batch).sum().item()
        accuracy = correct / tag_batch.size(0)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, accuracy: {accuracy:.2%}")
        w1, w2 = model.linear.weight[0]
        b = model.linear.bias[0]
        plot_x = torch.linspace(-2, 10, 50).view(-1, 1)
        plot_y = (-w1 * plot_x - b) / w2  # 计算决策边界
        plt.clf()  # 清除当前图形
        plt.xlim(-2, 10)
        plt.ylim(-2, 20)
        plt.scatter(x0.numpy(), y0.numpy(), color='blue', label='Category 0')
        plt.scatter(x1.numpy(), y1.numpy(), color='red', label='Category 1')
        plt.plot(plot_x.detach().numpy(), plot_y.detach().numpy(), color='black', label='Decision Boundary')
        plt.title('Binary Category Decision Boundary')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.pause(1.5)

plt.show(block = True)

        # 绘制当前模型决策边界点
        # with torch.no_grad():
        #     x_range = torch.linspace(-2, 10, 100).view(-1, 1)
        #     y_range = torch.linspace(-2, 10, 100).view(-1, 1)
        #     grid_x, grid_y = torch.meshgrid(x_range.squeeze(), y_range.squeeze())
        #     grid_points = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), dim=1)
        #     predictions = model(grid_points)
        #     predictions = predictions.reshape(grid_x.shape)

        #     plt.figure(figsize=(8, 6))
        #     plt.contour(grid_x.numpy(), grid_y.numpy(), predictions.numpy(), levels=[0.5], colors='b', linewidths=2)
        #     plt.scatter(x0.numpy(), y0.numpy(), color='blue', label='Category 0')
        #     plt.scatter(x1.numpy(), y1.numpy(), color='red', label='Category 1')
        #     plt.title('Binary Category Decision Boundary')
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.legend()
        #     plt.pause(0.5)
    

