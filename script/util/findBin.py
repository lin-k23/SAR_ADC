# import numpy as np
from math import gcd

def find_bin(Fs, Fin, N):
    bin_index = int(Fin / Fs * N)

    ptr_p = bin_index
    ptr_n = bin_index

    for _ in range(1000):
        ptr_p += 1
        ptr_n -= 1

        if gcd(N, ptr_p) == 1:
            return ptr_p

        if gcd(N, ptr_n) == 1:
            return ptr_n

    return bin_index  # Default return if not found
