import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    n=10000
    P = np.zeros((seq_len, d_model))
    # extend dimensions for broadcast
    pos = np.arange(seq_len)[:, np.newaxis]
    # range from 0 to d_model//2 - 1
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




# other variants
def get_angles(position, k, d_model):
    i = k // 2
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    return position * angle_rates

def positional_encoding2(positions, d_model): 
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis], 
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
  
    # apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding

