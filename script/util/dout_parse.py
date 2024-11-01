import numpy as np

"""
    从 d_out 中解析出 addr, ok 和 digital_code
    """


def dout_parse(d_out):
    # 获取 d_out 的尺寸
    L_data, n_dout_bits = d_out.shape

    # 确认 d_out 的位数为 16
    assert n_dout_bits == 16, "Expected 16-bit output data"

    # 解析 addr, ok 和 digital_code
    addr = np.dot(
        d_out[:, 0:3], [4, 2, 1]
    )  # 等效于 [d_out[:, 0]*4 + d_out[:, 1]*2 + d_out[:, 2]*1]
    ok = np.dot(d_out[:, 3:5], [2, 1])  # 等效于 [d_out[:, 3]*2 + d_out[:, 4]*1]
    digital_code = d_out[:, 5:16]  # 获取第 6 到第 16 位

    return addr, ok, digital_code
