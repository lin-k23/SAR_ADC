from preset_mdl import preset_mdl  # 导入 preset_mdl.py 中的函数
from preset_pr import preset_pr  # 导入 preset_pr.py 中的函数
from signal_source import signal_source  # 导入 signal_source.py 中的函数
from RISCA_core import RISCA_core  # 导入 RISCA_core.py 中的函数
from analyser.analyser_new import Analyser  # 导入 AnalyserSar 类
from util.load_assembler_xlsx import load_assembler_xlsx
import matplotlib.pyplot as plt  # 导入 matplotlib 库
import os

mdl = preset_mdl()
pr = preset_pr()
# pr["conf_name"] = input("sar/tisar/pipesar2s/nssar1o1c/nssar1o1ccp\n")
pr["conf_name"] = "nssar1o1c"
config_file_path = os.path.join("..\config", pr["conf_name"])
print(config_file_path)
pr_loaded = load_assembler_xlsx(config_file_path)
# update pr with pr_loaded
pr.update(pr_loaded)
if "T_assembler" in pr:
    print("Loaded Excel configuration:\n")
    print(pr["T_assembler"])

# Instantiate the device under test
da = RISCA_core(mdl, pr, signal_source(pr, mdl))
# analysis
test = Analyser(da, pr, mdl)
test.mode_analyser()
plt.show()
