### Hands on: fundations for Pytorch
* simple_linear_dynamic.py

简单的一元变量线性拟合样例，展示了基础的数据训练过程。
其中使用了 matplotlib 进行训练过程的动态可视化展现，更易于观察整个训练过程的渐进逼近。

笔记摘要：
1. nn.Linear
2. torch.linspace
3. squeeze / unsqueeze
4. torch.rand 及数据分布平移和缩放
5. matplotlib 动态绘图 fig.set_data + plt.pause
6. .detach.numpy()

效果示例：

![train1](imgs/simple_linear_dynamic_01.png)


