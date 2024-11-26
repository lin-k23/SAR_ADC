from preset_mdl import preset_mdl  # 导入 preset_mdl.py 中的函数
from preset_pr import preset_pr  # 导入 preset_pr.py 中的函数
from signal_source import signal_source  # 导入 signal_source.py 中的函数
from RISCA_core import RISCA_core  # 导入 RISCA_core.py 中的函数
from analyser.analyser import AnalyserSar  # 导入 AnalyserSar 类
from analyser.analyser_new import Analyser  # 导入 AnalyserSar 类
import pandas as pd  # 导入 pandas 库
import matplotlib.pyplot as plt  # 导入 matplotlib 库

# 调用函数获取模型参数
mdl = preset_mdl()
# print(mdl)

# 调用函数获取测试参数
pr = preset_pr()

# pr["conf_name"] = input("配置文件名: sar/tisar/nssar1o1c/noisar1o1ccp/pipesar2s/...\n")
print(pr["T_assembler"])

# 定义输入信号峰值
v_in_peak = [0.85, 0]
# 调用 signal_source 函数
v_in_p, v_in_n = signal_source(pr, mdl, v_in_peak)

# 以上无误

# Instantiate the device under test
da = RISCA_core(mdl, pr, v_in_p, v_in_n)

# analysis
test = Analyser(da, pr)
test.mode_analyser()

# py should use show while ipynb should not
plt.show()
