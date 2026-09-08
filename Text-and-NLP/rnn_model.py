import torch
import torch.nn as nn

class RNNUnit(nn.Module):
    """
    h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
    y_hat_t = W_hy * h_t + b_y
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True) # add bias
        
        self._init_weights()

    def _init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x_t, h_prev):
        # h_t = tanh(W_xh * x_t + W_hh * h_t-1 + b)
        out = self.W_xh(x_t) + self.W_hh(h_prev)
        h_t = torch.tanh(out)
        return h_t


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # support multi layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(RNNUnit(layer_input_size, hidden_size))

    def forward(self, x):
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
            # reshape to : (seq_len, batch_size, input_size)
            x = x.permute(1, 0, 2)
        else:
            seq_len, batch_size, _ = x.size()

        h_all = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        last_layer_outputs = []
        for t in range(seq_len):
            # current_input : (batch_size, input_size)
            current_input  = x[t]
            # layers loop         
            for layer_idx in range(self.num_layers):
                # each layer
                h_prev = h_all[layer_idx]  # shape: (batch, hidden_size)
                h_t = self.layers[layer_idx](current_input, h_prev)                
                h_all[layer_idx] = h_t

                current_input = h_t

            # last layer
            last_layer_outputs.append(current_input)
            
        # stack: (seq_len, batch, hidden_size)
        outputs = torch.stack(last_layer_outputs, dim=0)
        
        if self.batch_first:
            outputs = outputs.permute(1, 0, 2)
            
        # the last h_n : (num_layers, batch, hidden_size)
        h_n = h_all
        
        return outputs, h_n
