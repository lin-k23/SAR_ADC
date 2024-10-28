import os
import pandas as pd

def load_assembler_xlsx(path_config, conf_name):
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
    f_assembler = os.path.join(path_config, conf_name, f"{conf_name}_asm.xlsx")

    print(f"Loading file: {f_assembler}")

    # 检查文件是否存在
    if os.path.exists(f_assembler):
        # 读取 Excel 文件
        pr['T_assembler'] = pd.read_excel(f_assembler, index_col=0)  # index_col=0 用于将第一列作为行名
        # 把数据读打印放在主函数中
        # print(pr['T_assembler'])

        # 获取行数
        row = len(pr['T_assembler'].index)

        # 初始化 conf_n_CB 数组，初始值为 False
        conf_n_CB = [False] * row

        # 遍历行名，检查是否包含 'CB'
        for i in range(row):
            if 'CB' in pr['T_assembler'].index[i]:
                conf_n_CB[i] = True

        # 计算任务通道数量
        pr['task_ch'] = sum(conf_n_CB)

        # 获取 assembler 表的列数（task_frame）
        pr['task_frame'] = pr['T_assembler'].shape[1]

    else:
        raise FileNotFoundError(f"Config file not found: \n{f_assembler}")

    # 加载补丁文件部分的逻辑可以继续扩展
    pr['TI'] = 1

    return pr
