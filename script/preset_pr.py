# from util.load_assembler_xlsx import load_assembler_xlsx
import os
import pandas as pd


def load_assembler_xlsx(path_config):
    """
    从配置路径和配置名中加载 Excel 配置文件，并返回包含配置的字典。

    参数：
    - path_config: 配置文件所在的路径
    - conf_name: 配置文件的基本名称（不包括 "_asm.xlsx" 后缀）

    返回值：
    - 包含表数据和其他参数的字典 pr
    """
    pr = {}

    # 拼接 Excel 文件路径
    f_assembler = os.path.join(path_config, "timing_table.xlsx")

    print(f"Loading file: {f_assembler}")

    # 检查文件是否存在
    if os.path.exists(f_assembler):
        # 读取 Excel 文件
        pr["T_assembler"] = pd.read_excel(
            f_assembler, index_col=0
        )  # index_col=0 用于将第一列作为行名
        # 把数据读打印放在主函数中
        # print(pr['T_assembler'])

        # 获取行数
        row = len(pr["T_assembler"].index)

        # 初始化 conf_n_CB 数组，初始值为 False
        conf_n_CB = [False] * row

        # 遍历行名，检查是否包含 'CB'
        for i in range(row):
            if "CB" in pr["T_assembler"].index[i]:
                conf_n_CB[i] = True

        # 计算任务通道数量
        pr["task_ch"] = sum(conf_n_CB)

        # 获取 assembler 表的列数（task_frame）
        pr["task_frame"] = pr["T_assembler"].shape[1]

    else:
        raise FileNotFoundError(f"Config file not found")

    f_patch = os.path.join(path_config, "patch.xlsx")

    patch_data = pd.read_excel(f_patch)
    param_name = "TI"
    if param_name in patch_data["Item"].values:
        pr["TI"] = patch_data.loc[patch_data["Item"] == param_name, "Value"].values[0]
        print(f'[{__name__}] TI parameter loaded: pr["TI"]= {pr["TI"]}')
    else:
        print(f"[{__name__}] Warning: TI parameter not found in {f_patch}")
        pr["TI"] = 1

    return pr


def preset_pr():
    # Testbench parameters
    pr = {
        "test_type": "sim",  # Test type
        "OSR": 1,
        "TI": 1,
        "N_fft": 2**13,
        "N_ex_sam": 10,  # Number of extra samples
        "N_run": 1,  # Multiple run for spectrum averaging
        # Clock parameters
        "F_clk": 800e6,  # Frame clock frequency
        "Vpp_clk": 2,  # Frame clock Vpp
        # Input signal parameters for channel 1
        "source_type_1": "sim",  # Source type for ch1 [no / rigol / apx]
        "F_in_center_1": 99e6,  # Input frequency
        "F_in_split_1": 0,  # Input frequency splitting (=0 for single tone)
        "V_in_peak_1": 0,
        "V_in_os_1": 0,
        # Input signal parameters for channel 2
        "source_type_2": "notest",  # Source type for ch2 [no / rigol / apx]
        "F_in_center_2": 0,  # Input frequency
        "F_in_split_2": 0,  # Input frequency splitting (=0 for single tone)
        "V_in_peak_2": 0,
        "V_in_os_2": 0,
        "AVDD_scalling": 1,
        # Channel mapping
        "channel_mapping1": 1,
        "channel_mapping2": 2,
        "channel_mapping3": 3,
        "channel_mapping4": 4,
        "channel_mapping5": 5,
        "channel_mapping6": 6,
        # Configuration file name
        # "conf_name": input(
        #     "配置文件名: sar/tisar/nssar1o1c/noisar1o1ccp/pipesar2s/...\n"
        # ),
    }

    # # 把pr["conf_name"]和path_config拼接起来，得到配置文件的路径
    # path_config = "..\\config\\"
    # config_file_path = f"{path_config}{pr['conf_name']}\\"

    # # 调用 load_assembler_xlsx 函数加载配置文件
    # # pr_loaded = load_assembler_xlsx(path_config, pr["conf_name"], "timing_table")
    # pr_loaded = load_assembler_xlsx(config_file_path)

    # # update pr with pr_loaded
    # pr.update(pr_loaded)

    # 打印读取到的 Excel 配置文件
    # if "T_assembler" in pr:
    #     print("Loaded Excel configuration:")
    return pr


# test
if __name__ == "__main__":
    pr = preset_pr()
    print(type(pr["T_assembler"]))
    print("Loaded configuration:")
    print(f"Task channels: {pr['task_ch']}")
    print(f"Task frames: {pr['task_frame']}")
