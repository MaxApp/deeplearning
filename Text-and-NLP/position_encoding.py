import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    n=10000
    P = np.zeros((seq_len, d_model))
    # extend dimensions for broadcast
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model // 2)
    denominator = n ** (2 * i / d_model)
    # even positions
    P[:, 0::2] = np.sin(pos / denominator)
    # odd positions
    P[:, 1::2] = np.cos(pos / denominator)
    return P

if __name__ == "__main__":
    # sample encodings with 128 dimension
    pe = positional_encoding(100, 128)
    plt.title("Positional Encodings")
    plt.pcolormesh(pe[:, :], cmap='Blues')
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.colorbar()
    plt.show()
