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

y_data1 = [-20, -9.778, 4.944, 19.903, 31.613, 38.483, 53.749, ]  # 营业收入, y_data1[0]的IRR为"无"，设置为-20

y_data2 = [19.851, 19.877, 19.888, 19.903, 19.919, 19.929, 19.955, ]  # 固定和无形资产残值

y_data3 = [19.87, 19.888, 19.894, 19.90, 19.912, 19.918, 19.93, ]  # 回收流动资金（残值）

y_data4 = [22.62, 21.21, 20.68, 19.90, 19.16, 18.68, 17.54, ]  # 建设投资

y_data5 = [19.99, 19.95, 19.93, 19.90, 19.88, 19.86, 19.81, ]  # 流动资金

y_data6 = [52.34, 37.59, 31.01, 19.90, 6.01, -7.02, -20]  # 经营成本，y_data6[6]的IRR为"无"，设置为50

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
        color="#4962FC",
        markpoint_opts=opts.MarkPointOpts(
            data=[opts.MarkPointItem(name="该点IRR无法计算！", coord=[x_data[0], y_data1[0]], value="无")]
        ),  # 标注IRR无法计算的点
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
        color="#FD9E06",
        markpoint_opts=opts.MarkPointOpts(
            data=[opts.MarkPointItem(name="该点IRR无法计算！", coord=[x_data[6], y_data6[6]], value="无")]
        ),
    )
        .render("sensitivity_analysis-C.html")
)
