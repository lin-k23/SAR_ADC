from util.load_assembler_xlsx import load_assembler_xlsx

import pandas as pd


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
        "conf_name": "sar",
    }

    # 把pr["conf_name"]和path_config拼接起来，得到配置文件的路径
    path_config = "..\\config\\"
    config_file_path = f"{path_config}{pr['conf_name']}\\"

    # 调用 load_assembler_xlsx 函数加载配置文件
    # pr_loaded = load_assembler_xlsx(path_config, pr["conf_name"], "timing_table")
    pr_loaded = load_assembler_xlsx(config_file_path)

    # update pr with pr_loaded
    pr.update(pr_loaded)

    # 打印读取到的 Excel 配置文件
    if "T_assembler" in pr:
        print("Loaded Excel configuration:")
        # print(pr["T_assembler"])
        # data frame
    #     cfg_info = pd.DataFrame(pr["T_assembler"])
    #     styled_df = (
    #         cfg_info.style.set_table_attributes('class="dataframe table table-striped"')
    #         .set_precision(2)
    #         .highlight_max(color="lightgreen")
    #         .highlight_min(color="lightcoral")
    #     )
    # print(styled_df)
    # 返回组合后的参数字典
    return pr


# 调试
if __name__ == "__main__":
    pr = preset_pr()
    print(type(pr["T_assembler"]))
    print("Loaded configuration:")
    print(f"Task channels: {pr['task_ch']}")
    print(f"Task frames: {pr['task_frame']}")
