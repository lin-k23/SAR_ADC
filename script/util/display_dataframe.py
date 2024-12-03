from IPython.display import display, HTML


# 用于展示配置信息的函数
def display_dataframe(cfg_info, type):
    styled_df = (
        cfg_info.style.set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#1E3A8A"),  # 深蓝色背景
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("font-size", "14px"),  # 字体大小
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
                        ("font-size", "16px"),
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
                        ("font-size", "20px"),  # 标题字体大小
                    ],
                },  # 标题高亮
                # 设置表头字体粗细和颜色
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),  # 字体加粗
                        ("text-align", "center"),  # 居中显示
                    ],
                },  # 字体颜色
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),  # 居中显示
                    ],
                },  # 字体颜色
            ]
        )
        .set_table_attributes('class="dataframe table table-striped"')
        .set_properties(**{"text-align": "center"})
        .set_caption(
            f"Configuration Information for {type}",
        )
    )
    display(HTML("<div style='text-align: center"))
    display(styled_df)
    display(HTML("</div>"))
