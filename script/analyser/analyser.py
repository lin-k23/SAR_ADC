import numpy as np
import matplotlib.pyplot as plt
from util.specPlot import specPlot
from util.dout_parse import dout_parse

# class Analyser:
class AnalyserSar:
    def __init__(self, da, pr):
        self.da = da
        self.pr = pr
        self.weight_nom = np.array([256, 128, 64, 32, 16, 16, 8, 4, 4, 2, 1])

    def dout_parse(self):
        _, n_dout_bits = self.da.shape
        assert n_dout_bits == 16, "d_out should have 16 bits per row."
        addr = np.dot(self.da[:, :3], np.array([4, 2, 1]))
        ok = np.dot(self.da[:, 3:5], np.array([2, 1]))
        digital_code =self.da[:, 5:16]

        return addr, ok, digital_code

    def no_calibration(self):
        _, _, da = self.dout_parse()
        data_nocal = self.weight_nom @ da[:, :].T
        n_pts_plot = 1024

        # Plot da without calibration
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        plt.plot(data_nocal[:n_pts_plot], '-')
        plt.ylim([0, np.sum(self.weight_nom)])
        plt.xlim([0, n_pts_plot])
        plt.title('No Calibration')

        plt.subplot(2, 1, 2)
        ENoB, SNDR, SFDR, SNR, THD, pwr, NF, _ = specPlot(
            data_nocal.reshape(1, -1),
            self.pr['F_s'],
            np.sum(self.weight_nom)
        )

        offset_nocal = np.mean(data_nocal) - np.sum(self.weight_nom) / 2
        print(f'offset_nocal = {offset_nocal:.2f} LSB')

# Example usage:
# da = {'addr': ..., 'ok': ..., 'd_out': np.array(...)}
# analyser = AnalyserSar(da, pr)
# analyser.no_calibration()