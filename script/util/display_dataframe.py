from IPython.display import display, HTML


# 用于展示配置信息的函数
def display_dataframe(cfg_info, type):
    styled_df = (
        cfg_info.style.set_table_styles(
            [
                {
                    "selector": "thead th",
                    "display": "table-cell",
                    "width": "150px",
                    "props": [
                        ("background-color", "#1E3A8A"),  # 深蓝色背景
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("font-size", "26px"),  # 字体大小
                        ("text-align", "center"),
                        ("min-width", "150px"),
                    ],
                },  # 表头列宽
                {
                    "selector": "tbody td",
                    "props": [
                        ("background-color", "#F1F5F9"),  # 浅灰色背景
                        ("color", "#1F2937"),  # 深灰色字体
                        ("font-weight", "bold"),
                        ("font-size", "20px"),
                        ("min-width", "150px"),  # 设置列宽
                        ("max-width", "300px"),  # 最大宽度限制
                        ("height", "40px"),
                    ],
                },  # 增加行高
                {
                    "selector": "caption",
                    "props": [
                        ("font-weight", "bold"),  # 标题加粗
                        ("color", "white"),
                        ("font-size", "25px"),  # 标题字体大小
                    ],
                },  # 标题高亮
                # 设置表头字体粗细和颜色
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),  # 字体加粗
                        ("text-align", "center"),  # 居中显示
                    ],
                },
                # 设置表格边框
                {
                    "selector": "td",
                    "props": [
                        ("color", "#1F2937"),  # 深灰色字体
                        ("border", "3px solid #D1D5DB"),  # 单元格边框
                    ],
                },
            ]
        )
        .set_table_attributes('class="dataframe table table-striped"')
        .set_properties(**{"text-align": "center"})
        .set_caption(
            f"Timing Table",
        )
    )
    display(HTML("<div style='text-align: center"))
    display(styled_df)
    display(HTML("</div>"))


def get_config_info(conf_name, pr):
    pr["conf_name"] = conf_name
    print(f"config file: {conf_name}")
