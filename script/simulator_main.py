from preset_mdl import preset_mdl  # 导入 preset_mdl.py 中的函数
from preset_pr import preset_pr  # 导入 preset_pr.py 中的函数
from signal_source import signal_source  # 导入 signal_source.py 中的函数
from RISCA_core import RISCA_core  # 导入 RISCA_core.py 中的函数
from analyser.analyser_new import Analyser  # 导入 AnalyserSar 类
from util.load_assembler_xlsx import load_assembler_xlsx
import matplotlib.pyplot as plt  # 导入 matplotlib 库
import os
import pandas as pd

# 调用函数获取模型参数
mdl = preset_mdl()
# 调用函数获取测试参数
pr = preset_pr()
pr["conf_name"] = input("sar/tisar/pipesar2s/nssar1o1c/nssar1o1ccp\n")
config_file_path = os.path.join("../config", pr["conf_name"])
print(config_file_path)
pr_loaded = load_assembler_xlsx(config_file_path)
# update pr with pr_loaded
pr.update(pr_loaded)
if "T_assembler" in pr:
    print("Loaded Excel configuration:\n")
    print(pr["T_assembler"])
if pr["conf_name"] == "tisar":
    mdl["n_wgt_sar1"] = [256, 128, 64, 32, 16]
    mdl["n_wgt_sar2"] = [16, 8, 4, 2, 1, 0.5]

# 定义输入信号峰值
v_in_peak = [pr["v_in_peak"], 0]
# 调用 signal_source 函数
v_in_p, v_in_n = signal_source(pr, mdl, v_in_peak)

# Instantiate the device under test
da = RISCA_core(mdl, pr, v_in_p, v_in_n)
# da_read = pd.read_csv("da.csv")
# da = da_read.to_numpy()
# analysis
test = Analyser(da, pr)
test.mode_analyser()

# py should use show while ipynb should not
plt.show()
