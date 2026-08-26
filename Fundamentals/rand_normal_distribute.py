
"""
生成均匀分布和正态分布数据，并绘制直方图
"""

import torch
import matplotlib.pyplot as plt

# 生成数据
uniform_data = torch.rand(10000)  # [0,1)均匀分布
normal_data = torch.randn(10000)  # 标准正态分布

# 创建画布
plt.figure(figsize=(12, 5))

# 绘制均匀分布
plt.subplot(1, 2, 1)
plt.hist(uniform_data.numpy(), bins=50, color='skyblue', edgecolor='black')
plt.title('torch.rand() - Uniform Distribution')
plt.xlabel('Value Range [0,1)')
plt.ylabel('Frequency')

# 绘制正态分布
plt.subplot(1, 2, 2)
plt.hist(normal_data.numpy(), bins=50, color='salmon', edgecolor='black')
plt.title('torch.randn() - Normal Distribution')
plt.xlabel('Value Range (-∞,+∞)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
