"""
# 简单线性手动梯度下降示例

模型 y = w1*x1+w2*x2+b
数据 [[x1, x2, y]]
目标参数 w1 = 2, w2 = 3, b = 5
"""

import random
import matplotlib.pyplot as plt

# [x1, x2, y]
data = [
    [0, 0, 5],  
    [1, 0, 7],  
    [0, 1, 8],  
    [1, 1, 10],  
    [2, 0, 9],  
    [0, 2, 11],  
    [2, 1, 12],  
    [1, 2, 13],  
    [3, 0, 11],  
    [0, 3, 14],  
    [3, 1, 14],  
    [1, 3, 16],  
    [4, 0, 13],  
    [0, 4, 17],  
    [2, 2, 15]  
]

# 产生随机种子
random.seed(42)
# 随机生成 w1, w2, b
w1 = random.uniform(0, 10)
w2 = random.uniform(0, 10)
b = random.uniform(0, 10)
print(f"Initial weights: w1={w1}, w2={w2}, b={b}")

# 学习率
lr = 0.001

# 前向传播函数
def forward(x1, x2):
    return w1 * x1 + w2 * x2 + b

# 损失函数
def loss_func(y_pred, y_true):
    return (y_pred - y_true) ** 2

# 迭代次数
epochs = 10000
# 记录迭代次数和MSE
epoch_losses = []
# 训练模型
for epoch in range(1,epochs+1):
    # 随机从 data 中取样，样本数为 10
    sample_data = random.sample(data, 10)
    # 打印样本数据
    # print("Sampled data:")
    # for x1, x2, y_true in sample_data:
    #     print(f"x1: {x1}, x2: {x2}, y_true: {y_true}")

    total_loss = 0.0
    grad_w1 = 0.0
    grad_w2 = 0.0
    grad_b = 0.0
    
    for x1, x2, y_true in sample_data:
        # 前向传播
        y_pred = forward(x1, x2)
        
        # 累计批量损失
        loss = loss_func(y_pred, y_true)
        total_loss += loss
        
        # 累计批量梯度
        grad_w1 += 2 * x1 * (y_pred - y_true)
        grad_w2 += 2 * x2 * (y_pred - y_true)
        grad_b += 2 * (y_pred - y_true)
    
    # 小批量更新权重和偏置
    w1 -= lr * (grad_w1 / len(sample_data))
    w2 -= lr * (grad_w2 / len(sample_data))
    b -= lr * (grad_b / len(sample_data))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Average Loss: {total_loss / len(sample_data)}, w1: {w1}, w2: {w2}, b: {b}")
    
    if epoch > 2000 and epoch % 100 == 0:
        # 记录一组 epoch 和MSE
        epoch_losses.append((epoch, total_loss / len(sample_data)))

# 绘制损失曲线
epochs, losses = zip(*epoch_losses)
plt.plot(epochs, losses, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training Loss Curve')
plt.grid()
plt.show()






