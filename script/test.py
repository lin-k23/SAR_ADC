# gen a sin
import numpy as np
import matplotlib.pyplot as plt
from util.specPlot import specPlot
from preset_pr import preset_pr
from preset_mdl import preset_mdl
from util.findBin import find_bin


def gen_sin(f, fs, n):
    n = 2**15
    t = np.arange(0, n) / fs
    wave = np.sin(2 * np.pi * f * t)
    # add noise
    noise = np.random.normal(0, 1e-3, n)
    wave = wave + noise
    return wave


pr = preset_pr()
mdl = preset_mdl()
pr["F_s"] = pr["F_clk"] / 3  # Sampling frequency for each input
pr["n_in_bin_1"] = find_bin(pr["F_s"], pr["F_in_center_1"], pr["N_fft"])

# pr["n_in_bin_1"] = 1
pr["n_in_bin_2"] = find_bin(pr["F_s"], pr["F_in_center_2"], pr["N_fft"])
# Calculate input frequencies
pr["F_in"] = np.array([pr["n_in_bin_1"], pr["n_in_bin_2"]]) / pr["N_fft"] * pr["F_s"]
f = pr["F_in"][0]
t = 1e-6
fs = pr["F_s"]
wave = gen_sin(f, fs, t)

plt.figure()
plt.subplot(2, 1, 1)


plt.plot(wave)

sin_matrix = wave.reshape(1, -1).T

plt.subplot(2, 1, 2)
specPlot(sin_matrix, fs)
plt.show()
