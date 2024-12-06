from preset_mdl import preset_mdl
from preset_pr import preset_pr
from signal_source import signal_source
from RISCA_core import RISCA_core
from analyser.analyser_new import Analyser
from util.load_assembler_xlsx import load_assembler_xlsx
import matplotlib.pyplot as plt
import os

mdl = preset_mdl()
pr = preset_pr()
# pr["conf_name"] = input("sar/tisar/pipesar2s/nssar1o1c/nssar1o1ccp\n")
pr["conf_name"] = "pipesar3s"
config_file_path = os.path.join("..\config", pr["conf_name"])
print(config_file_path)
pr_loaded = load_assembler_xlsx(config_file_path)
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
