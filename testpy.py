# 谐波合成
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3 * np.pi, 3 * np.pi, 10000)
y1 = np.sin(x)
y2 = np.sin(3 * x) * 1 / 3
y3 = np.sin(5 * x) * 1 / 5
y4 = np.sin(7 * x) * 1 / 7
y5 = np.sin(9 * x) * 1 / 9
y = y1 + y2 + y3 + y4 + y5

# 绘制图像 两幅子图 1行2列 一行显示所有原始信号  一行显示合成信号
plt.figure(figsize=(12, 6))

# Plot original signals
plt.subplot(1, 2, 1)
plt.plot(x, y1, label="sin(x)")
plt.plot(x, y2, label="sin(3x)/3")
plt.plot(x, y3, label="sin(5x)/5")
plt.plot(x, y4, label="sin(7x)/7")
plt.plot(x, y5, label="sin(9x)/9")
plt.title("Original Signals")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# Plot combined signal
plt.subplot(1, 2, 2)
plt.plot(x, y, label="Combined Signal")
plt.title("Combined Signal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
