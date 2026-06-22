import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

NUM_OF_DATA = 100
torch.manual_seed(24)
x_true = torch.linspace(1,100,NUM_OF_DATA).unsqueeze(1)
# 增加 noise
noise = (torch.rand(NUM_OF_DATA,1) * 2 - 1) * 0.18
# y 符合 log10 分布
y_true = torch.log10(x_true) + noise

# model definition
# 特别注意：
# 由于输入数据 x, y 的范围不一致，x ∈ (1,100), y ∈ (0,2)
# 如果不进行 normalization，则训练中造成梯度消失
model = nn.Sequential(
    nn.BatchNorm1d(1), # data normalization 的重要性
    nn.Linear(1, 5),
    nn.ReLU(),
    nn.BatchNorm1d(5),
    nn.Linear(5,10),
    nn.ReLU(),
    nn.BatchNorm1d(10),
    nn.Linear(10,1)
)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

# 绘制 true data
axs[0].set_title("Model Fitting")
axs[0].scatter(x_true, y_true)
plot_line, = axs[0].plot([],[], c="r")

# Loss function
loss_function = nn.MSELoss()
# optimizer
# 使用 Adam 明显收敛更快速
optimizer = optim.Adam(model.parameters(), lr=0.02)

# 记录 loss 变化值
LOSS_RECORDS = 60
loss_values = {"sgd":{"epochs":[], "losses":[]}, "adam":{"epochs":[], "losses":[]}}

epochs = 1000
for epoch in range(epochs):
    # 清空初始化梯度
    optimizer.zero_grad()
    y_hat = model(x_true)
    loss = loss_function(y_hat, y_true)
    loss.backward()
    # update parameter
    optimizer.step()

    if epoch <= LOSS_RECORDS:
        loss_values["adam"]["epochs"].append(epoch)
        loss_values["adam"]["losses"].append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"epoch:{epoch+1} Loss:{loss.item()}")
        plot_line.set_data(x_true.detach().numpy(), y_hat.detach().numpy())
        plt.pause(0.8)  # 短暂暂停以显示更新

    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         grad_mean = param.grad.abs().mean().item()
    #         grad_max = param.grad.abs().max().item()
    #         print(f"{name:12s} | grad_mean: {grad_mean:.2e} | grad_max: {grad_max:.2e}")

# SGD 性能对照
model2 = nn.Sequential(
    nn.BatchNorm1d(1), 
    nn.Linear(1, 5),
    nn.ReLU(),
    nn.BatchNorm1d(5),
    nn.Linear(5,10),
    nn.ReLU(),
    nn.BatchNorm1d(10),
    nn.Linear(10,1)
)
optimizerSGD = optim.SGD(model2.parameters(), lr=0.02)

for epoch in range(epochs):
    # 清空初始化梯度
    optimizerSGD.zero_grad()
    y_hat = model2(x_true)
    loss = loss_function(y_hat, y_true)
    loss.backward()
    # update parameter
    optimizerSGD.step()

    if epoch <= LOSS_RECORDS:
        loss_values["sgd"]["epochs"].append(epoch)
        loss_values["sgd"]["losses"].append(loss.item())

axs[1].plot(loss_values["adam"]["epochs"], loss_values["adam"]["losses"], c="g", label="Adam")
axs[1].plot(loss_values["sgd"]["epochs"], loss_values["sgd"]["losses"], c="y", label="SGD")
axs[1].legend()
axs[1].set_title("Loss Curve")
plt.show()

