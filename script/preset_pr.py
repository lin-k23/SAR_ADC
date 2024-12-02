# from util.load_assembler_xlsx import load_assembler_xlsx
import os
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
        "v_in_peak": 0.85,
    }
    return pr


# test
if __name__ == "__main__":
    pr = preset_pr()
    print(type(pr["T_assembler"]))
    print("Loaded configuration:")
    print(f"Task channels: {pr['task_ch']}")
    print(f"Task frames: {pr['task_frame']}")
