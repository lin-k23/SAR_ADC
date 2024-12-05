# load fft data

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("tdata.csv")[1:]
result = (np.fft.fft(data)) ** 2
print(result)
