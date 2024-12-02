import numpy as np
import matplotlib.pyplot as plt
from util.specPlot import specPlot
from util.specPlot import specPlotOS


# class Analyser:
class Analyser:
    """
    Analyser class
    """

    def __init__(self, da, pr):
        self.mode = pr["conf_name"]
        self.da = da
        self.pr = pr
        self.mode_analyser = self._get_mode_analyser()

    def dout_parse(self):
        _, n_dout_bits = self.da.shape
        assert n_dout_bits == 16, "d_out should have 16 bits per row."
        # i_run=1;
        addr = np.dot(self.da[:, :3], np.array([4, 2, 1]))
        ok = np.dot(self.da[:, 3:5], np.array([2, 1]))
        digital_code = self.da[:, 5:16]

        return addr, ok, digital_code

    def plot_triangle(self, order, st_x, st_y, ratio_x, color="b"):
        """
        Plot a triangle in a given plot.

        Parameters:
        - order: The order of the triangle (affects height scaling).
        - st_x: Starting x-coordinate.
        - st_y: Starting y-coordinate.
        - ratio_x: Ratio for scaling x-coordinates.
        - color: Color of the triangle lines (default: 'b' for blue).
        """
        # Calculate triangle vertices
        x1, y1 = st_x * ratio_x, st_y
        x2, y2 = x1, st_y + 20 * order * np.log10(ratio_x)
        x3, y3 = st_x, st_y

        # Draw the triangle edges
        plt.plot([x1, x2], [y1, y2], color, linewidth=2)
        plt.plot([st_x, x1], [st_y, st_y], color, linewidth=2)
        plt.plot([st_x, x1], [st_y, y2], color, linewidth=2)

    def _get_mode_analyser(self):
        mode_mappinng = {
            "sar": self._sar_analyser,
            "tisar": self._tisar_analyser,
            "nssar1o1c": self._nssar1o1c_analyser,
            "noisar1o1ccp": self._noisar1o1ccp_analyser,
            "pipesar2s": self._pipesar2s_analyser,
        }
        if self.mode in mode_mappinng:
            return mode_mappinng[self.mode]
        else:
            raise ValueError(f"Mode {self.mode} not supported.")

    def _sar_analyser(self):
        """
        Analyser for SAR ADC
        """
        self.weight_nom = np.array([256, 128, 64, 32, 16, 16, 8, 4, 4, 2, 1])

        n_pts_plot = 1024
        _, _, da_1 = self.dout_parse()
        da = da_1[-self.pr["N_fft"] :, :]
        data_nocal = self.weight_nom @ da[:, :].T

        # Plot da without calibration
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        plt.plot(data_nocal[:n_pts_plot], "-")
        plt.ylim([0, np.sum(self.weight_nom)])
        plt.xlim([0, n_pts_plot])
        plt.title("No Calibration")
        # for debug
        # plt.show()

        plt.subplot(2, 1, 2)
        # ENoB, SNDR, SFDR, SNR, THD, pwr, NF, _ = specPlot(
        #     data_nocal.reshape(1, -1).T, self.pr["F_s"], np.sum(self.weight_nom)
        # )
        _, _, _, _, _, _, _, _ = specPlot(
            data_nocal.reshape(1, -1).T, self.pr["F_s"], np.sum(self.weight_nom)
        )
        # ---------------------------------------
        # if use jupyter notebook, not use plt.show()
        # plt.show()
        # ---------------------------------------
        offset_nocal = np.mean(data_nocal) - np.sum(self.weight_nom) / 2
        print(f"offset_nocal = {offset_nocal:.2f} LSB")

    def _tisar_analyser(self):
        """
        Analyser for TISAR ADC
        """
        #  weight
        weight1 = np.array([256, 128, 64, 32, 16, 16, 8, 4, 2, 1, 0.5])
        weight2 = np.array([256, 128, 64, 32, 16, 16, 8, 4, 2, 1, 0.5])
        weight3 = np.array([256, 128, 64, 32, 16, 16, 8, 4, 2, 1, 0.5])
        # data parse and alignment
        _, _, digital_code = self.dout_parse()
        # N_run = 1
        pr_N_fft = self.pr["N_fft"]
        data1 = np.zeros((pr_N_fft, len(weight1)))
        data2 = np.zeros((pr_N_fft, len(weight2)))
        data3 = np.zeros((pr_N_fft, len(weight3)))

        N_out = 3
        data1[:, :] = digital_code[-pr_N_fft * 3 + 0 :: N_out, :]
        data2[:, :] = digital_code[-pr_N_fft * 3 + 1 :: N_out, :]
        data3[:, :] = digital_code[-pr_N_fft * 3 + 2 :: N_out, :]
        #  load weight, gain
        data_cal = np.zeros((pr_N_fft * 3))
        # data1_cal = np.dot(weight1, data1[:].T)
        data1_cal = weight1 @ data1[:, :].T
        std1 = np.std(data1_cal)
        data1_cal = data1_cal - np.mean(data1_cal)
        # data2_cal = np.dot(weight2, data2[:pr_N_fft, :].T)
        data2_cal = weight2 @ data2[:, :].T
        data2_cal = (data2_cal - np.mean(data2_cal)) / np.std(data2_cal) * std1
        # data3_cal = np.dot(weight3, data3[:pr_N_fft, :].T)
        data3_cal = weight3 @ data3[:, :].T
        data3_cal = (data3_cal - np.mean(data3_cal)) / np.std(data3_cal) * std1

        data_comb_run = np.concatenate([data1_cal, data2_cal, data3_cal], axis=0)
        data_cal[:] = data_comb_run[: pr_N_fft * 3] + np.sum(weight1) / 2

        # Calculate metrics using a Python equivalent of specPlot
        ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = specPlot(
            data_cal.reshape(1, -1).T, self.pr["F_s"], np.sum(weight1)
        )

        #  power compensation and FOM
        da_Power = 1e-3
        OSR = 1  # Replace with your actual OSR values
        BW = self.pr["F_s"] / 2 / OSR
        FoMs = SNDR + 10 * np.log10(self.pr["F_s"] / 2 / da_Power)
        FoMw = (da_Power) / (self.pr["F_s"] / (2**ENoB)) * 1e15

        print(
            f"[{__file__}] ENoB={ENoB:.2f}, SNDR={SNDR:.1f}dB, SFDR={SFDR:.1f}dB, "
            f"FoMs={FoMs:.1f}dB, FoMw={FoMw:.1f}fJ, BW={BW/1e6:.1f}M, Power={da_Power*1e3:.2f}mW"
        )

    def _nssar1o1c_analyser(self):
        """
        Analyser for NSSAR1O1C ADC
        """
        addr, ok, digital_code = self.dout_parse()
        weight_nom = np.array([256, 128, 64, 32, 16, 16, 8, 4, 4, 2, 1])
        OSR = 32
        data = digital_code[-self.pr["N_fft"] :, :]
        aout = weight_nom @ data.T
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        n_pts_plot = self.pr["N_fft"] // 1
        plt.plot(aout[:n_pts_plot])
        plt.ylim([0, np.sum(weight_nom)])
        plt.xlim([0, n_pts_plot])
        plt.title("Output Waveform")
        plt.xlabel("Samlpe Index")
        plt.ylabel("Amplitude")

        plt.subplot(1, 2, 2)
        ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = specPlotOS(
            aout.reshape(1, -1).T, self.pr["F_s"], np.sum(weight_nom), 5, "OSR", OSR
        )
        self.plot_triangle(1, 8e6, -100, 3)

    def _noisar1o1ccp_analyser(self):
        """
        Analyser for NOISAR1O1CCP ADC
        """
        addr, ok, digital_code = self.dout_parse()
        weight_nom = np.array([256, 128, 64, 32, 16, 16, 8, 4, 4, 2, 1])
        OSR = 32

        data_rec_1 = digital_code[1 + 10 : self.pr["N_fft"] + 10 : 2, :]
        data_rec_2 = digital_code[2 + 10 : self.pr["N_fft"] + 10 : 2, :]

        # 初始化 data_nocal
        data_nocal = np.zeros((self.pr["N_run"], self.pr["N_fft"]))

        # 计算 data_nocal
        for i_run in range(self.pr["N_run"]):
            aout1 = np.dot(weight_nom, data_rec_1[:, :, i_run].T)
            aout2 = np.dot(weight_nom, data_rec_2[:, :, i_run].T)
            data_comb = np.concatenate([aout1, np.sum(weight_nom) - aout2])
            # 转置并添加偏置
            data_nocal[i_run, :] = data_comb + np.sum(weight_nom) / 2
        # 可视化部分
        is_report = 0  # 示例值
        if not "is_report" in locals() or is_report == 0:
            # 创建绘图窗口
            fig = plt.figure(figsize=(15, 8))
            ax1 = fig.add_subplot(1, 2, 1)
        ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = specPlotOS(
            data_nocal.reshape(1, -1).T,
            self.pr["N_fft"],
            self.pr["F_s"],
            np.sum(weight_nom),
            5,
            "OSR",
            OSR,
        )

    def _pipesar2s_analyser(self):
        pass
