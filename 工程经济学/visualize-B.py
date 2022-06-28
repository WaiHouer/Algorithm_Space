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

y_data1 = [-2.121, 9.595, 13.660, 19.318, 24.573, 27.901, 35.740]  # 营业收入

y_data2 = [19.260, 19.289, 19.300, 19.318, 19.335, 19.346, 19.375]  # 固定和无形资产残值

y_data3 = [19.260, 19.289, 19.300, 19.318, 19.335, 19.346, 19.375]  # 回收流动资金（残值）

y_data4 = [21.910, 20.567, 20.057, 19.318, 18.608, 18.150, 17.055]  # 建设投资

y_data5 = [19.49, 19.41, 19.37, 19.32, 19.26, 19.23, 19.14, ]  # 流动资金

y_data6 = [33.941, 26.886, 23.931, 19.318, 14.438, 10.998, 1.478]  # 经营成本

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
        .render("sensitivity_analysis-B.html")
)
