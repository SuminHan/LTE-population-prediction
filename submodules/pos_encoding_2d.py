import numpy as np

def get_ae_pos_encoding():
    pos_encoding = np.load('../cid_encoding.npy')
    return pos_encoding.astype(np.float32)[18:18+32, 22:22+32, ...]

def get_area_encoding():
    pos_encoding = np.load('../road-to-embedding/area_vectors.npy')
    return pos_encoding.astype(np.float32)[18:18+32, 22:22+32, ...]

def get_2d_pos_encoding(row, col, d_model):
    # Compute the angle for each position and dimension
    pos_rows = np.arange(row)[:, np.newaxis]
    pos_cols = np.arange(col)[:, np.newaxis]
    angle_rads_row = pos_rows / np.power(10000, (2 * (np.arange(d_model // 2))) / np.float32(d_model))
    angle_rads_col = pos_cols / np.power(10000, (2 * (np.arange(d_model // 2))) / np.float32(d_model))

    # Apply sine to even indices in the array and cosine to odd indices
    pos_encoding = np.zeros((row, col, d_model), dtype=np.float32)
    pos_encoding[:, :, 0::2] = np.sin(angle_rads_row[:, np.newaxis, :] + angle_rads_col[np.newaxis, :, :])
    pos_encoding[:, :, 1::2] = np.cos(angle_rads_row[:, np.newaxis, :] + angle_rads_col[np.newaxis, :, :])

    return pos_encoding.astype(np.float32)


def get_2d_onehot_encoding(row, col):
    num_cells = row*col
    pos_encoding = np.zeros((row, col, num_cells), dtype=np.float32)
    for j in range(row):
        for i in range(col):
            pos_encoding[j, i, j*col + i] = 1
    return pos_encoding.astype(np.float32)
