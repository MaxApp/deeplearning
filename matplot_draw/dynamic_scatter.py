import numpy as np
import matplotlib.pyplot as plt

# 动态绘制散点图

plt.figure()
plt.xlim(0, 10)
plt.ylim(0, 10)

for i in range(10):
    plt.cla()  # 清除当前坐标轴（保留坐标系设置）
    plt.xlim(0, 10)  # 重新设置x轴范围（清除后会重置）
    plt.ylim(0, 10)  # 重新设置y轴范围
    x, y = np.random.rand(2) * 10
    plt.scatter(x, y, c='blue')
    plt.draw()
    plt.pause(0.5)

plt.show(block=True)