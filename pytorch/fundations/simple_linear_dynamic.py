import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# 用例：根据 x -> y , 预测简单的一元变量线性关系

# 样本数据
# Model : y = w * x + b
w = 0.4
b = 1.5
NUM_OF_DATA = 50
torch.manual_seed(40)
x_true = torch.linspace(0,1,NUM_OF_DATA).unsqueeze(1)
# print(f"=== { w * x_true + b}")
# 增加 noise
noise = (torch.rand(NUM_OF_DATA,1) * 2 - 1) * 0.08
# print(f"=== { noise}")
y_true = w * x_true + b + noise

# print(f"x: {x_true}")
# print(f"y: {y_true}")

# 绘制 true data
plt.scatter(x_true, y_true)
plot_line, = plt.plot([],[], c="r")

model = nn.Sequential(
    nn.Linear(1,1) # input 1d, output 1d
)

# Loss function
loss_function = nn.MSELoss()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.02)

epochs = 1000
for epoch in range(epochs):
    # 清空初始化梯度
    optimizer.zero_grad()
    y_hat = model(x_true)
    loss = loss_function(y_hat, y_true)
    loss.backward()
    # update parameter
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"epoch:{epoch+1} Loss:{loss.item()}")
        plot_line.set_data(x_true.detach().numpy(), y_hat.detach().numpy())
        plt.pause(0.8)  # 短暂暂停以显示更新

plt.show()
    