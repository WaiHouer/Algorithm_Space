"""
工程经济学大作业：敏感性分析代码-蜘蛛图

date:05-04

author：@王锦烨、陈典
"""
import pandas as pd
import datetime
from pyecharts import options as opts
from pyecharts.charts import Line, Timeline
import os
from pyecharts.globals import ThemeType
import random


x_data = ["-10%", "-5%", "-3%", "0%", "3%", "5%", "10%"]  # 横轴

y_data1 = [-2.477, 8.681, 12.592, 18.071, 23.196, 26.457, 34.188]  # 营业收入

y_data2 = [18.008, 18.040, 18.052, 18.071, 18.091, 18.103, 18.135]  # 固定和无形资产残值

y_data3 = [18.048, 18.060, 18.064, 18.071, 18.079, 18.083, 18.095]  # 回收流动资金（残值）

y_data4 = [20.602, 19.295, 18.796, 18.071, 17.374, 16.923, 15.842]  # 建设投资

y_data5 = [18.150, 18.111, 18.095, 18.071, 18.048, 18.032, 17.993, ]  # 流动资金

y_data6 = [32.574, 25.513, 22.591, 18.071, 13.344, 10.045, 1.050]  # 经营成本

(
    Line(init_opts=opts.InitOpts(theme=ThemeType.CHALK,
                                 animation_opts=opts.AnimationOpts(animation_delay_update=5800,
                                                                   animation_delay=1800,
                                                                   animation_duration=5000,
                                                                   animation_duration_update=5500,
                                                                   animation_easing_update="cubicOut",
                                                                   )))
        .set_global_opts(

        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        title_opts=opts.TitleOpts(title="敏感性分析", pos_left=400, padding=[30, 20])  # 标题居中
    )

        .add_xaxis(xaxis_data=x_data)
        .add_yaxis(
        series_name="营业收入",
        y_axis=y_data1,
        symbol="emptyCircle",
        is_symbol_show=True,
        is_smooth=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2),
        color="#4962FC"
    )
        .add_yaxis(
        series_name="固定和无形资产残值",
        y_axis=y_data2,
        symbol="emptyCircle",
        is_symbol_show=True,
        is_smooth=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2),
        color="#4B7CF3"
    )
        .add_yaxis(
        series_name="回收流动资金（残值）",
        y_axis=y_data3,
        symbol="emptyCircle",
        is_symbol_show=True,
        is_smooth=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2),
        color="#dd3ee5"
    )
        .add_yaxis(
        series_name="建设投资",
        y_axis=y_data4,
        symbol="emptyCircle",
        is_symbol_show=True,
        is_smooth=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2),
        color="#12e78c"
    )

        .add_yaxis(
        series_name="流动资金",
        y_axis=y_data5,
        symbol="emptyCircle",
        is_symbol_show=True,
        is_smooth=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2),
        color="#fe8104"
    )
        .add_yaxis(
        series_name="经营成本",
        y_axis=y_data6,
        symbol="emptyCircle",
        is_symbol_show=True,
        is_smooth=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2),
        color="#FD9E06"
    )
        .render("sensitivity_analysis-A.html")
)
