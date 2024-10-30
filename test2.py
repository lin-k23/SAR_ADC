# FFT of a signal
import numpy as np
import matplotlib.pyplot as plt

fs = 4000  # Sampling frequency
Ts = 1 / fs  # Sampling period
N = 1000  # Number of samples
f_in1 = 100  # Input frequency 1
f_in2 = 200  # Input frequency 2
t = np.arange(0, N * Ts, Ts)  # Time vector
X_n = np.sin(2 * np.pi * f_in1 * t) + 2 * np.sin(2 * np.pi * f_in2 * t)  # Signal
Y = np.fft.fft(X_n)

# Plot the magnitude spectrum as discrete impulses
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t, X_n)
plt.xlabel("t (s)")
plt.ylabel("X_n(t)")
plt.title("Signal")
plt.grid()

# Create frequency vector
f = np.fft.fftfreq(N, Ts)

plt.subplot(1, 2, 2)
plt.stem(f[: N // 2], np.abs(Y)[: N // 2], basefmt=" ")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum (Discrete)")
peak_freq = f[np.argmax(np.abs(Y[: N // 2]))]
plt.text(
    1,
    1,
    f"Peak Frequency={peak_freq:.2f}Hz",
    fontsize=12,
    ha="right",
    va="top",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="none", edgecolor="red", boxstyle="round,pad=0.5"),
)
# 用红框显示最高峰值的频率
plt.axvline(x=peak_freq, color="red", linestyle="--")
plt.grid()

plt.tight_layout()
plt.show()
