import numpy as np
import matplotlib.pyplot as plt
from util.specPlot import specPlot
from util.specPlot import specPlotOS
from PIL import Image


# class Analyser:
class Analyser:
    """
    Analyser class
    """

    def __init__(self, da, pr, mdl):
        self.mode = pr["conf_name"]
        self.da = da
        self.pr = pr
        self.mdl = mdl
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
            "nssar1o1ccp": self._noisar1o1ccp_analyser,
            "pipesar2s": self._pipesar2s_analyser,
            "pipesar3shp": self._pipesar3shp_analyser,
        }
        if self.mode in mode_mappinng:
            return mode_mappinng[self.mode]
        else:
            raise ValueError(f"Mode {self.mode} not supported.")

    def _sar_analyser(self):
        """
        Analyser for SAR ADC
        """
        self.weight_nom = np.array(self.mdl["n_wgt_sar1"] + self.mdl["n_wgt_sar2"])

        n_pts_plot = 16
        _, _, da_1 = self.dout_parse()
        da = da_1[-self.pr["N_fft"] - 2 : -2, :]
        data_nocal = self.weight_nom @ da[:, :].T

        # Plot da without calibration
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        img = Image.open("..\pic\\nssar1o1c.png")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.plot(data_nocal[:n_pts_plot], "-o")
        plt.ylim([0, np.sum(self.weight_nom)])
        plt.xlim([0, n_pts_plot])
        plt.title("No Calibration")

        plt.subplot(1, 3, 3)
        _, _, _, _, _, _, _, _ = specPlot(
            data_nocal.reshape(1, -1).T, self.pr["F_s"], np.sum(self.weight_nom)
        )
        offset_nocal = np.mean(data_nocal) - np.sum(self.weight_nom) / 2
        print(f"offset_nocal = {offset_nocal:.2f} LSB")

    def _tisar_analyser(self):
        """
        Analyser for TISAR ADC
        """
        #  weight
        weight1 = np.array(self.mdl["n_wgt_sar1"] + self.mdl["n_wgt_sar2"])
        weight2 = np.array(self.mdl["n_wgt_sar1"] + self.mdl["n_wgt_sar2"])
        weight3 = np.array(self.mdl["n_wgt_sar1"] + self.mdl["n_wgt_sar2"])
        # data parse and alignment
        _, _, digital_code = self.dout_parse()
        # N_run = 1
        pr_N_fft = self.pr["N_fft"]
        data1 = np.zeros((pr_N_fft, len(weight1)))
        data2 = np.zeros((pr_N_fft, len(weight2)))
        data3 = np.zeros((pr_N_fft, len(weight3)))

        N_out = 3
        data1[:, :] = digital_code[-pr_N_fft * N_out + 0 : -2 : N_out, :]
        data2[:, :] = digital_code[-pr_N_fft * N_out + 1 : -1 : N_out, :]
        data3[:, :] = digital_code[-pr_N_fft * N_out + 2 :: N_out, :]
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

        # data_comb_run = np.concatenate([data1_cal, data2_cal, data3_cal], axis=0)
        data_comb_run = np.empty((pr_N_fft * 3,))
        data_comb_run[0::3] = data1_cal
        data_comb_run[1::3] = data2_cal
        data_comb_run[2::3] = data3_cal
        data_cal[:] = data_comb_run[: pr_N_fft * 3] + np.sum(weight1) / 2

        # Plot da without calibration
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        img = Image.open("..\pic\\tisar.png")
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        n_pts_plot = pr_N_fft // 1
        plt.plot(data_cal[:n_pts_plot])
        plt.ylim([0, np.sum(weight1)])
        plt.xlim([0, n_pts_plot])
        plt.title("No Calibration")
        plt.subplot(1, 3, 3)
        # Calculate metrics using a Python equivalent of specPlot
        ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = specPlot(
            data_cal.reshape(1, -1).T, self.pr["F_s"], np.sum(weight1)
        )

    def _nssar1o1c_analyser(self):
        """
        Analyser for NSSAR1O1C ADC
        """
        _, _, digital_code = self.dout_parse()
        weight_nom = np.array(self.mdl["n_wgt_sar1"] + self.mdl["n_wgt_sar2"])
        OSR = 32
        data = digital_code[-self.pr["N_fft"] :, :]
        aout = weight_nom @ data.T

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        img = Image.open("..\pic\\nssar1o1c.png")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        n_pts_plot = self.pr["N_fft"] // 1
        plt.plot(aout[:n_pts_plot])
        plt.ylim([0, np.sum(weight_nom)])
        plt.xlim([0, n_pts_plot])

        plt.subplot(1, 3, 3)
        _, _, _, _, _, _, _, _ = specPlotOS(
            aout.reshape(1, -1),
            self.pr["N_fft"],
            self.pr["F_s"],
            np.sum(weight_nom),
            harmonic=5,
            OSR=OSR,
        )
        # self.plot_triangle(1, 8e6, -100, 3)

    def _noisar1o1ccp_analyser(self):
        """
        Analyser for NOISAR1O1CCP ADC
        """
        _, _, digital_code = self.dout_parse()
        weight_nom = np.array(self.mdl["n_wgt_sar1"] + self.mdl["n_wgt_sar2"])
        OSR = 32

        data_rec_1 = digital_code[10 : self.pr["N_fft"] + 10 : 2, :]
        data_rec_2 = digital_code[1 + 10 : self.pr["N_fft"] + 10 : 2, :]

        # 初始化 data_nocal
        data_comb = np.zeros(self.pr["N_fft"])
        data_comb1 = np.zeros(self.pr["N_fft"])
        data_nocal = np.zeros(self.pr["N_fft"])
        # 计算 data_nocal
        aout1 = weight_nom @ data_rec_1[:, :].T
        aout2 = weight_nom @ data_rec_2[:, :].T
        data_comb[0::2] = aout1
        data_comb[1::2] = np.sum(weight_nom) - aout2
        data_nocal[:] = data_comb + np.sum(weight_nom) / 2

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        img = Image.open("../pic/nssar1o1ccp.png")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        n_pts_plot = self.pr["N_fft"] // 1
        data_comb1[0::2] = aout1
        data_comb1[1::2] = aout2
        plt.plot(data_comb[:n_pts_plot])
        plt.ylim([0, np.sum(weight_nom)])
        plt.xlim([0, n_pts_plot])
        # 创建绘图窗口
        plt.subplot(1, 3, 3)
        plt.title("No Calibration")
        _, _, _, _, _, _, _, _ = specPlotOS(
            data_nocal.reshape(1, -1),
            self.pr["N_fft"],
            self.pr["F_s"],
            np.sum(weight_nom),
            harmonic=5,
            OSR=OSR,
        )

    def _pipesar2s_analyser(self):
        addr, _, digital_code = self.dout_parse()
        digital_code = digital_code[-self.pr["N_fft"] :, :]

        ## para
        weight_nom_1 = np.array(self.mdl["n_wgt_sar1"] + [0, 0, 0, 0, 0, 0])
        weight_nom_2 = np.array(self.mdl["n_wgt_sar1"] + self.mdl["n_wgt_sar2"])
        out_sequence = np.array([1, 2])
        shift_sequence = np.array([0, 1])
        kb_include = np.array([1])
        seg_check = np.array([[1, 0], [0, 1], [8, 1]])
        weight_nom = np.array([weight_nom_1, weight_nom_2])
        n_stage = 2
        n_bits = len(weight_nom_1) + len(weight_nom_2)
        n_seg_check = 3
        n_stage2 = 2
        n_stage_bits = np.array([len(weight_nom_1), len(weight_nom_2)])
        n_out = 2

        ## alignment
        data_comb = np.zeros([self.pr["N_fft"], np.sum(n_stage_bits)])
        data_comb_kb = np.zeros(
            [
                self.pr["N_fft"],
                np.sum(n_stage_bits) + np.sum(n_stage_bits[kb_include - 1]),
            ]
        )
        st = 2
        ed = n_out * self.pr["N_fft"] + st - 1
        shift = 0
        for i1 in range(n_out):
            if addr[st - 1] == self.pr["channel_mapping" + str(i1 + 1)]:
                shift = n_out - (np.where(out_sequence == i1 + 1)[0][0] + 1)
            tmp = []
            tmp2 = []
            for i2 in range(n_stage):
                data = digital_code[
                    st
                    + shift
                    + shift_sequence[i2] : ed
                    + 1
                    + shift
                    + shift_sequence[i2] : n_out,
                    :,
                ]
                tmp.append(data[:, -int(n_stage_bits[i2]) :])
            tmp2 = [np.copy(arr) for arr in tmp]
            for i3 in range(len(kb_include)):
                data_d = digital_code[
                    st
                    + shift
                    + shift_sequence[kb_include[i3]]
                    - 1
                    - n_out : ed
                    + shift
                    + shift_sequence[kb_include[i3]]
                    - n_out : n_out,
                    :,
                ]
                tmp2.append(data_d[:, -n_stage_bits[kb_include[i3]] :])

            data_comb[:, :] = np.hstack(tmp)
            data_comb_kb[:, :] = np.hstack(tmp2)

        ## calibration
        n_pts_plot = self.pr["N_fft"] // 64
        plt.figure(figsize=(15, 6))

        for i1 in range(n_seg_check):
            weight_seg = []
            for i2 in range(n_stage):
                weight_seg.append(seg_check[i1, i2] * weight_nom[i2, :])
            weight_seg = np.array(weight_seg)

            seg_index = seg_check[i1, :]

            p_first_nonzero = 0
            p_last_nonzero = 0

            for ix in range(n_stage2):
                if seg_index[ix] > 0:
                    p_first_nonzero = ix
                    break

            for ix in range(n_stage2 - 1, 0, -1):
                if seg_index[ix] > 0:
                    p_last_nonzero = ix
                    break

            st = 0
            ed = 0
            for ix in range(p_first_nonzero):
                st += n_stage_bits[ix]
            st += 1
            for ix in range(p_last_nonzero + 1):
                ed += n_stage_bits[ix]

            data_seg = data_comb[:, st - 1 : ed]
            aout_seg = weight_seg @ data_seg.T

            plt.subplot(2, n_seg_check, i1 + 1)
            plt.plot(aout_seg[:n_pts_plot])
            plt.ylim([0, np.sum(weight_seg)])
            plt.xlim([0, n_pts_plot])

            seg_check_str = f"seg check: [{', '.join(map(str, seg_check[i1, :]))}]"
            plt.title(seg_check_str)
            plt.subplot(2, n_seg_check, n_seg_check + i1 + 1)
            ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = specPlot(
                aout_seg, self.pr["F_s"], np.sum(weight_seg)
            )

    def _pipesar3shp_analyser(self):

        pass
