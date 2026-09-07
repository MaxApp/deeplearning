import numpy as np

class RNNCell:
    """
    h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        
        # 用于存储前向传播过程中的中间变量，以便反向传播使用
        self.cache = {}

    def forward(self, x, h_prev):
        """ 
        返回:
            h_next (np.array): 当前时间步的隐藏状态，形状 (hidden_size, 1)
        """
        # 计算线性组合: z = W_xh * x + W_hh * h_prev + b_h
        z = np.dot(self.W_xh, x) + np.dot(self.W_hh, h_prev) + self.b_h
        
        # 应用激活函数 tanh
        h_next = np.tanh(z)
        
        # 缓存中间结果用于反向传播
        self.cache['x'] = x
        self.cache['h_prev'] = h_prev
        self.cache['h_next'] = h_next
        self.cache['z'] = z
        
        return h_next

    def backward(self, dh_next):
        """
        单个时间步的反向传播 (BPTT 的一步)。
        
        参数:
            dh_next (np.array): 来自下一层或下一时间步的关于 h_next 的梯度，形状 (hidden_size, 1)
            
        返回:
            dx (np.array): 关于输入 x 的梯度
            dh_prev (np.array): 关于上一时刻隐藏状态 h_prev 的梯度
            grads (dict): 包含 dW_xh, dW_hh, db_h 的字典
        """
        # 从缓存中获取前向传播的数据
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        h_next = self.cache['h_next']
        z = self.cache['z']
        
        # 1. tanh 激活函数的导数: dtanh(z) = 1 - tanh(z)^2
        dtanh_z = 1 - h_next ** 2
        
        # 2. 计算经过激活函数前的梯度 dz
        dz = dh_next * dtanh_z
        
        # 3. 计算权重的梯度
        dW_xh = np.dot(dz, x.T)
        dW_hh = np.dot(dz, h_prev.T)
        db_h = dz
        
        # 4. 计算输入的梯度和上一时刻隐藏状态的梯度
        dx = np.dot(self.W_xh.T, dz)
        dh_prev = np.dot(self.W_hh.T, dz)
        
        # 保存梯度
        grads = {
            'dW_xh': dW_xh,
            'dW_hh': dW_hh,
            'db_h': db_h
        }
        
        return dx, dh_prev, grads


class RNNSequence:
    
    def __init__(self, input_size, hidden_size):
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward_sequence(self, inputs, h0=None):

        if h0 is None:
            h0 = np.zeros((self.hidden_size, 1))
            
        h_prev = h0
        hidden_states = []
        
        for t, x_t in enumerate(inputs):
            h_next = self.cell.forward(x_t, h_prev)
            hidden_states.append(h_next)
            h_prev = h_next
            
        return hidden_states, h_prev

    def backward_sequence(self, hidden_states, inputs, dh_final):
        """
        对整个序列进行反向传播 (BPTT)。
        
        参数:
            hidden_states (list): 前向传播得到的隐藏状态列表
            inputs (list): 输入数据列表
            dh_final (np.array): 最终隐藏状态的梯度
            
        返回:
            total_grads (dict): 累积的梯度
        """
        # 初始化累积梯度
        total_grads = {
            'dW_xh': np.zeros_like(self.cell.W_xh),
            'dW_hh': np.zeros_like(self.cell.W_hh),
            'db_h': np.zeros_like(self.cell.b_h)
        }
        
        dh_next = dh_final
        
        # 从最后一个时间步反向遍历到第一个时间步
        for t in reversed(range(len(inputs))):
            # 恢复该时间步的缓存数据 (注意：StandardRNNCell 的 cache 会被覆盖，
            # 在实际复杂实现中，通常需要在 forward_sequence 中保存所有时间步的 cache)
            # 这里为了简化演示，我们假设 cell 能够访问对应时间步的数据，
            # 或者更严谨的做法是修改 StandardRNNCell 使其支持批量缓存。
            
            # *修正*：由于上面的 StandardRNNCell 只保留最后一步的 cache，
            # 真正的 BPTT 需要每一步的 cache。为了代码简洁且可运行，
            # 我们在这里重新模拟一次前向传播来填充 cache，或者修改类结构。
            # 下面采用一种更通用的方式：直接在 RNNSequence 中管理所有步骤的局部梯度计算。
            
            x_t = inputs[t]
            h_prev = np.zeros((self.hidden_size, 1)) if t == 0 else hidden_states[t-1]
            h_next = hidden_states[t]
            
            # 手动执行单步反向逻辑 (复用 StandardRNNCell 的逻辑思想)
            z = np.dot(self.cell.W_xh, x_t) + np.dot(self.cell.W_hh, h_prev) + self.cell.b_h
            dtanh_z = 1 - h_next ** 2
            dz = dh_next * dtanh_z
            
            total_grads['dW_xh'] += np.dot(dz, x_t.T)
            total_grads['dW_hh'] += np.dot(dz, h_prev.T)
            total_grads['db_h'] += dz
            
            dh_next = np.dot(self.cell.W_hh.T, dz) # 传递给前一个时间步
            
        return total_grads

if __name__ == "__main__":
  
    input_size = 3
    hidden_size = 4
    seq_len = 5
    
    rnn_seq = RNNSequence(input_size, hidden_size)
    
    # 生成随机输入序列: 列表包含 seq_len 个形状为 (input_size, 1) 的向量
    inputs = [np.random.randn(input_size, 1) for _ in range(seq_len)]
    
    hidden_states, h_final = rnn_seq.forward_sequence(inputs)
    print(f"最终隐藏状态形状: {h_final.shape}")
    print(f"隐藏状态值示例:\n{h_final.flatten()[:5]}")
    
    # 反向传播测试
    # 假设损失函数对最终隐藏状态的梯度是全 1 向量
    dh_final = np.ones((hidden_size, 1))
    grads = rnn_seq.backward_sequence(hidden_states, inputs, dh_final)
    
    print("\n梯度检查:")
    print(f"dW_xh 形状: {grads['dW_xh'].shape}")
    print(f"dW_hh 形状: {grads['dW_hh'].shape}")
    print(f"db_h 形状: {grads['db_h'].shape}")
