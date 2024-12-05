import numpy as np
from util.findBin import find_bin
import matplotlib.pyplot as plt


def signal_source(pr, mdl):
    # Calculate sampling frequency
    # mdl['is_verbose'] = 1
    v_in_peak = [pr["v_in_peak"], 0]
    pr["F_s"] = (
        pr["F_clk"] / pr["task_frame"] * pr["TI"]
    )  # Sampling frequency for each input

    # Determine input frequency bins
    pr["n_in_bin_1"] = find_bin(pr["F_s"], pr["F_in_center_1"], pr["N_fft"])
    pr["n_in_bin_2"] = find_bin(pr["F_s"], pr["F_in_center_2"], pr["N_fft"])

    # Calculate input frequencies
    pr["F_in"] = (
        np.array([pr["n_in_bin_1"], pr["n_in_bin_2"]]) / pr["N_fft"] * pr["F_s"]
    )

    if mdl["is_verbose"] == 0:
        n_sim = pr["N_fft"] * pr["task_frame"] + pr["N_ex_sam"]  # simulation points
    else:
        n_sim = 10

    t_sim = np.arange(n_sim) / pr["F_clk"]  # absolute time of each sim point

    v_in_p = np.zeros((mdl["n_in_bus"], n_sim))
    v_in_n = np.zeros((mdl["n_in_bus"], n_sim))

    # Input signal
    for iter_in in range(mdl["n_in_bus"]):
        v_in_p[iter_in, :] = (
            v_in_peak[iter_in] / 2 * np.cos(pr["F_in"][iter_in] * t_sim * 2 * np.pi)
        )
        v_in_n[iter_in, :] = (
            -v_in_peak[iter_in] / 2 * np.cos(pr["F_in"][iter_in] * t_sim * 2 * np.pi)
        )

    # Plot input signals
    if mdl["is_verbose"] == 1:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t_sim * 1e6, v_in_p[0, :], label="Ch1")
        plt.plot(t_sim * 1e6, v_in_p[1, :], label="Ch2")
        plt.xlabel("Time (us)")
        plt.ylabel("Voltage (V)")
        plt.title("Positive Input Signal")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t_sim * 1e6, v_in_n[0, :], label="Ch1")
        plt.plot(t_sim * 1e6, v_in_n[1, :], label="Ch2")
        plt.xlabel("Time (us)")
        plt.ylabel("Voltage (V)")
        plt.title("Negative Input Signal")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return v_in_p, v_in_n
