import os
import pandas as pd


def load_assembler_xlsx(path_config):
    pr = {}
    f_assembler = os.path.join(path_config, "timing_table.xlsx")
    conf_table = os.path.join(path_config, "timing_table_1.xlsx")
    pr["conf_table"] = pd.read_excel(conf_table, index_col=0)
    # print(f"Loading file: {f_assembler}")
    if os.path.exists(f_assembler):
        pr["T_assembler"] = pd.read_excel(f_assembler, index_col=0)

        row = len(pr["T_assembler"].index)

        conf_n_CB = [False] * row

        for i in range(row):
            if "CB" in pr["T_assembler"].index[i]:
                conf_n_CB[i] = True
        pr["task_ch"] = sum(conf_n_CB)
        pr["task_frame"] = pr["T_assembler"].shape[1]

    else:
        raise FileNotFoundError(f"Config file not found")

    # load patch
    f_patch = os.path.join(path_config, "patch.xlsx")

    if os.path.exists(f_patch):
        patch_data = pd.read_excel(f_patch)
        for x in patch_data["Item"].values:
            pr[x] = patch_data.loc[patch_data["Item"] == x, "Value"].values[0]
    return pr
