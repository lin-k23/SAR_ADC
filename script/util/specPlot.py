import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fftpack import fft


# alias ok
def alias(J, N):
    if (J * 2 // N) % 2 == 0:
        bin_value = J - (J // N) * N
    else:
        bin_value = N - J + (J // N) * N
    return bin_value


def specPlot(
    data,
    Fs=1,
    maxCode=None,
    harmonic=7,
    winType=np.hanning,
    sideBin=1,
    logSca=0,
    label=1,
    assumedSignal=np.nan,
    isPlot=1,
):

    N, M = (data).shape

    if maxCode is None:
        maxCode = np.max(np.max(data)) - np.min(np.min(data))

    Nd2 = N // 2
    freq = np.arange(Nd2) / N * Fs
    win = winType(N)

    spec = np.zeros(N)
    ME = 0

    for iter in range(M):
        tdata = data[:, iter]
        if np.sqrt(np.mean(tdata**2)) == 0:  # RMS = 0
            continue
        tdata = tdata / maxCode
        tdata = tdata - np.mean(tdata)
        tdata = tdata * win / np.sqrt(np.mean(win**2))
        spec += np.abs(np.fft.fft(tdata)) ** 2
        ME += 1

    spec = spec[:Nd2]
    spec[0] = 0  # DC分量设置为0
    spec = spec / (N**2) * 16 / ME  # 归一化

    bin_max = np.argmax(spec)
    bin_max = 2440
    sig = np.sum(spec[max(bin_max - sideBin, 0) : min(bin_max + sideBin, Nd2)])
    pwr = 10 * np.log10(sig)

    if not np.isnan(assumedSignal):
        sig = 10 ** (assumedSignal / 10)
        pwr = assumedSignal

    # 处理谐波
    if harmonic < 0:
        for i in range(2, -harmonic + 1):
            b = alias((bin_max) * i, N)
            spec[max(b + 1 - sideBin, 0) : min(b + 1 + sideBin, Nd2)] = 0

    if isPlot:
        # avoid log(0)
        spec[spec == 0] = 10 ** (-20)

        if logSca == 0:
            plt.plot(freq, 10 * np.log10(spec))
        else:
            plt.semilogx(freq, 10 * np.log10(spec))

        plt.grid(True)
        if label:
            if logSca == 0:
                plt.plot(
                    freq[max(bin_max - sideBin, 0) : min(bin_max + sideBin, Nd2)],
                    10
                    * np.log10(
                        spec[max(bin_max - sideBin, 0) : min(bin_max + sideBin, Nd2)]
                    ),
                    "r-",
                    linewidth=0.5,
                )
            else:
                plt.semilogx(
                    freq[max(bin_max - sideBin, 0) : min(bin_max + sideBin, Nd2)],
                    10
                    * np.log10(
                        spec[max(bin_max - sideBin, 0) : min(bin_max + sideBin, Nd2)]
                    ),
                    "r-",
                    linewidth=0.5,
                )

        if harmonic > 0:
            for i in range(2, harmonic + 1):
                b = alias((bin_max) * i, N)
                plt.plot(b / N * Fs, 10 * np.log10(spec[b + 1] + 10 ** (-20)), "rs")
                plt.text(
                    b / N * Fs,
                    10 * np.log10(spec[b + 1] + 10 ** (-20)) + 5,
                    str(i),
                    fontsize=12,
                    ha="center",
                )

    spec[max(bin_max - sideBin, 0) : min(bin_max + sideBin, Nd2) + 1] = 0
    noi = np.sum(spec)

    sbin = np.argmax(spec)
    spur = np.sum(spec[max(sbin - 5, 0) : min(sbin + 5, Nd2)])

    SNDR = 10 * np.log10(sig / noi)
    SFDR = 10 * np.log10(sig / spur)
    ENoB = (SNDR - 1.76) / 6.02

    thd = 0
    for i in range(2, N // 100):
        b = alias((bin_max) * i, N)
        thd += np.sum(spec[max(b + 1 - 2, 0) : min(b + 1 + 2, Nd2)])
        spec[max(b + 1 - 2, 0) : min(b + 1 + 2, Nd2)] = 0

    noi = np.sum(spec)
    THD = 10 * np.log10(thd / sig)
    SNR = 10 * np.log10(sig / noi)
    NF = SNR - pwr

    if isPlot:
        plt.axis([Fs / N, Fs / 2, min(max(-NF - 10 * np.log(N) - 10, -150), -40), 0])
        if label:
            plt.axvline(Fs / 2, linestyle="--")
            text_position = Fs / N * 2
            if Fs > 10**6:
                plt.text(
                    text_position,
                    -10,
                    f"Fin/Fs = {bin_max/N*Fs/10**6:.1f} / {Fs/10**6:.1f} MHz",
                )
            else:
                plt.text(
                    text_position,
                    -10,
                    f"Fin/Fs = {bin_max/N*Fs/10**3:.1f} / {Fs/10**3:.1f} KHz",
                )
            plt.text(bin_max / N * Fs, pwr, f"Fund = {pwr:.2f} dB")
            plt.text(text_position, -20, f"ENoB = {ENoB:.2f}")
            plt.text(text_position, -30, f"SNDR = {SNDR:.2f} dB")
            plt.text(text_position, -40, f"SFDR = {SFDR:.2f} dB")
            plt.text(text_position, -50, f"THD = {THD:.2f} dB")
            plt.text(text_position, -60, f"SNR = {SNR:.2f} dB")
            plt.text(text_position, -70, f"Noise Floor = {NF:.2f} dB")
            plt.xlabel("Freq (Hz)")
            plt.ylabel("dBFS")
            plt.title("Output Spectrum")
            # plt.show()

    return ENoB, SNDR, SFDR, SNR, THD, pwr, NF, None if not isPlot else plt.gca()


def specPlotOS(
    data,
    N_fft=1,
    Fs=1,
    maxCode=None,
    harmonic=5,
    OSR=1,
    winType="hann",
    sideBin=1,
    label=1,
    assumedSignal=None,
    isPlot=1,
    noFlicker=0,
):
    """
    Compute and optionally plot the output spectrum, signal metrics.

    Parameters:
    - data: 2D array where rows are individual runs of data.
    - N_fft: FFT size.
    - Fs: Sampling frequency.
    - maxCode: Maximum signal code value (default: max(data) - min(data)).
    - harmonic: Number of harmonics to remove.
    - OSR: Oversampling ratio.
    - winType: Window type, e.g., 'hann'.
    - sideBin: Number of bins to include around signal peak.
    - label: Whether to label the plot.
    - assumedSignal: Assumed signal power (dB).
    - isPlot: Whether to plot the results.
    - noFlicker: Remove flicker noise if set to 1.

    Returns:
    - ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h (plot handle).
    """
    if maxCode is None:
        maxCode = np.max(data) - np.min(data)

    N_run, _ = data.shape
    Nd2 = N_fft // 2
    freq = np.linspace(0, Fs / 2, Nd2)
    win = get_window(winType, N_fft, fftbins=True)

    spec = np.zeros(N_fft)
    ME = 0
    for tdata in data:
        if np.sqrt(np.mean(tdata**2)) == 0:
            continue
        tdata = tdata / maxCode
        tdata = tdata - np.mean(tdata)
        tdata = tdata * win / np.sqrt(np.mean(win**2))
        spec += np.abs(fft(tdata)) ** 2
        ME += 1

    spec = spec[:Nd2]
    spec[0] = 0
    spec = spec / (N_fft**2) * 16 / ME

    if noFlicker == 1:
        spec[: int(N_fft / 1e3)] = 0

    bin_idx = np.argmax(spec[: Nd2 // OSR])
    sig = np.sum(spec[max(bin_idx - sideBin, 0) : min(bin_idx + sideBin + 1, Nd2)])
    pwr = 10 * np.log10(sig)
    if assumedSignal is not None:
        sig = 10 ** (assumedSignal / 10)
        pwr = assumedSignal

    if harmonic < 0:
        for i in range(2, -harmonic + 1):
            b = (bin_idx * i) % N_fft
            spec[max(b - sideBin, 0) : min(b + sideBin + 1, Nd2)] = 0

    if isPlot:
        plt.figure()
        if OSR == 1:
            (h,) = plt.plot(freq, 10 * np.log10(spec))
        else:
            (h,) = plt.semilogx(freq, 10 * np.log10(spec))
        plt.grid()
        plt.title("Output Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dBFS")

        if label:
            plt.plot(
                freq[max(bin_idx - sideBin, 0) : min(bin_idx + sideBin + 1, Nd2)],
                10
                * np.log10(
                    spec[max(bin_idx - sideBin, 0) : min(bin_idx + sideBin + 1, Nd2)]
                ),
                "r-",
                linewidth=0.5,
            )
            plt.text(freq[bin_idx], pwr, f"Fund = {pwr:.2f} dB", fontsize=10)
    else:
        h = None

    # Remove signal components for noise calculation
    spec[max(bin_idx - sideBin, 0) : min(bin_idx + sideBin + 1, Nd2)] = 0
    noi = np.sum(spec[: Nd2 // OSR])

    spur = np.max(spec[bin_idx : Nd2 // OSR])
    SNDR = 10 * np.log10(sig / noi)
    SFDR = 10 * np.log10(sig / spur)
    ENoB = (SNDR - 1.76) / 6.02

    thd = 0
    for i in range(2, N_fft // 100 + 1):
        b = (bin_idx * i) % N_fft
        thd += np.sum(spec[max(b - 2, 0) : min(b + 3, Nd2 // OSR)])
        spec[max(b - 2, 0) : min(b + 3, Nd2 // OSR)] = 0

    noi = np.sum(spec[: Nd2 // OSR])
    THD = 10 * np.log10(thd / sig)
    SNR = 10 * np.log10(sig / noi)
    NF = SNR - pwr

    if isPlot:
        plt.axis([Fs / N_fft, Fs / 2, -150, 0])
        if OSR > 1:
            plt.text(Fs / 2, -80, f"OSR = {OSR}")
        plt.show()

    return ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h
