# 导入时间模块
from datetime import datetime, timezone
print(f"Last updated: {datetime.now(tz=timezone.utc):%d %B %Y %H:%M:%S %Z}")    # 格式化当前时间

import re
from datetime import datetime


# 导入模块  设置显示plt的图片 pandas显示最大列数12
import numpy as np
import pandas as pd


pd.options.display.max_columns = 12  # 最大列数
# 可视化模块
import plotly.graph_objects as go  # graph_objects 带日期的散点图
from plotly.subplots import make_subplots  # 用于绘制子图
from IPython.display import display # ipthon输出方法display


# 正则表达式\d数字{} 匹配1位到2位
date_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2}")


# 格式化日期为列名
# 1/22/20 转成22/01/2020
def reformat_dates(col_name: str) -> str:
    # for columns which are dates, I'd much rather they were in day/month/year format
    try:
        return date_pattern.sub(datetime.strptime(col_name, "%m/%d/%y").strftime("%d/%m/%Y"), col_name, count=1)
    except ValueError:
        return col_name

# 本地数据链接地址
confirmed_cases_url='time_series_covid19_confirmed_global.csv'
deaths_url='time_series_covid19_deaths_global.csv'

# 重命名列名字典
renamed_columns_map = {
    "Country/Region": "country",
    "Province/State": "location",
    "Lat": "latitude",
    "Long": "longitude"
}

# 要去掉的列 列表
cols_to_drop = ["location", "latitude", "longitude"]

# 读取csv文件重命名列名和去掉列，格式化一列的日期格式
# 确诊病例
confirmed_cases_df = (
    pd.read_csv(confirmed_cases_url)
    .rename(columns=renamed_columns_map)
    .rename(columns=reformat_dates)
    .drop(columns=cols_to_drop)
)
# 死亡病例
deaths_df = (
    pd.read_csv(deaths_url)
    .rename(columns=renamed_columns_map)
    .rename(columns=reformat_dates)
    .drop(columns=cols_to_drop)
)


# 打印表格默认的前后5行
display(confirmed_cases_df.head())
display(deaths_df.head())

# 国家名称去重复 即国家列表
geo_data_df = confirmed_cases_df[["country"]].drop_duplicates()
# 国家字母代号alpha-3_code 设置国家名称为索引index 如中国是CHN
country_codes_df = (
    pd.read_csv(
        "country_code_mapping.csv",
        usecols=["country", "alpha-3_code"],
        index_col="country")
)
# 把确诊病例 和 国家表合并 设置索引为国家名称
# 可以理解为 提取确诊病例国家代码
geo_data_df = geo_data_df.join(country_codes_df, how="left", on="country").set_index("country")

# 代码为空 并且 index不在 ["Diamond Princess", "MS Zaandam", "West Bank and Gaza"] 里面
# 结果为空的
geo_data_df[(pd.isnull(geo_data_df["alpha-3_code"])) & (~geo_data_df.index.isin(
    ["Diamond Princess", "MS Zaandam", "West Bank and Gaza"]
))]

# pandas 正则过滤不符合 xx/xx/xxxx 列名的列axis=1即Y轴 列的列名转成列表

'''
['22/01/2020',........ '12/06/2020']
'''
dates_list = (
    deaths_df.filter(regex=r"(\d{2}/\d{2}/\d{4})", axis=1)
        .columns
        .to_list()
)


# 映射 国家每日病例数和死亡病例数
# 以字典形式 country  confirmed_cases  deaths alpha-3_code
# key：是日期   值是：国家名称  确诊病例 死亡病例 国家代码 df格式
cases_by_date = {}
for date in dates_list:
    # 确诊病例日数量
    confirmed_cases_day_df = (
        confirmed_cases_df
            .filter(like=date, axis=1) # 顾虑不符合这个日期的列
            .rename(columns=lambda col: "confirmed_cases") # 匿名函数
    )
    # 死亡病例日数量
    deaths_day_df = deaths_df.filter(like=date, axis=1)\
        .rename(columns=lambda col: "deaths")
    # 两个df合并 设置国家为索引
    cases_df = confirmed_cases_day_df.join(deaths_day_df).set_index(confirmed_cases_df["country"])
    # 所有国家的合成一个表 分组按照国家名称 agg聚合 即计算总和sum alpha-3_code取第一行的值
    date_df = (
        geo_data_df.join(cases_df)
            .groupby("country")
            .agg({"confirmed_cases": "sum", "deaths": "sum", "alpha-3_code": "first"})
    )
    date_df = date_df[date_df["confirmed_cases"] > 0].reset_index()
    # 赋值给字典
    cases_by_date[date] = date_df
print(cases_by_date)


# 最后一个日期的数据
cases_by_date[dates_list[-1]].head()


# 显示世界地图
# 返回散点图数据参数格式 字典格式
def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


# 子图2行1列 样式scattergeo  这里是plotly画图模板数据，套进去即可画图
# 配置一些参数 标题
fig = make_subplots(rows=2, cols=1, specs=[[{"type": "scattergeo"}], [{"type": "xy"}]], row_heights=[0.8, 0.2])

fig.layout.geo = {"showcountries": True}
fig.layout.sliders = [{"active": 0, "steps": []}]
# 0-100 播放更新
fig.layout.updatemenus = [
    {
        "type": "buttons",
        "buttons": [
            {
                "label": "&#9654;",  # play symbol
                "method": "animate",
                "args": [None, frame_args(100)], #参数
            },
            {
                "label": "&#9724;",
                "method": "animate",  # stop symbol
                "args": [[None], frame_args(0)],
            },
        ],
        "showactive": False,
        "direction": "left",
    }
]
fig.layout.title = {"text": "Covid-19 Global Case Tracker", "x": 0.5}


frames = []
steps = []

# 设置确诊病例量程
max_country_confirmed_cases = cases_by_date[dates_list[-1]]["confirmed_cases"].max()

# 格式化为对数
high_tick = np.log1p(max_country_confirmed_cases)
low_tick = np.log1p(1)
log_tick_values = np.geomspace(low_tick, high_tick, num=6)

visual_tick_values = np.expm1(log_tick_values).astype(int)
visual_tick_values[-1] = max_country_confirmed_cases
visual_tick_values = [f"{val:,}" for val in visual_tick_values]

# 统计死亡比例总数 确诊比例总数 400k
# 第0个元素 是确诊比例 第1个元素  是死亡比例
cases_deaths_totals = [(df.filter(like="confirmed_cases").astype("uint32").agg("sum")[0],
                        df.filter(like="deaths").astype("uint32").agg("sum")[0])
                       for df in cases_by_date.values()]

confirmed_cases_totals = [daily_total[0] for daily_total in cases_deaths_totals]
deaths_totals = [daily_total[1] for daily_total in cases_deaths_totals]

# for循环枚举 开始1 制作好所有播放的帧图
for i, (date, data) in enumerate(cases_by_date.items(), start=1):
    df = data
    df["confirmed_cases_log"] = np.log1p(df["confirmed_cases"])
    # 鼠标经过国家位置 显示信息
    df["text"] = (
            date
            + "<br>"
            + df["country"]
            + "<br>Confirmed cases: "
            + df["confirmed_cases"].apply(lambda x: "{:,}".format(x)) # 格式化数字千进制
            + "<br>Deaths: "
            + df["deaths"].apply(lambda x: "{:,}".format(x)) # 格式化数字千进制
    )

    choro_trace = go.Choropleth(
        **{
            "locations": df["alpha-3_code"],
            "z": df["confirmed_cases_log"],
            "zmax": high_tick,
            "zmin": low_tick,
            "colorscale": "reds",
            "colorbar": {
                "ticks": "outside",
                "ticktext": visual_tick_values,
                "tickmode": "array",
                "tickvals": log_tick_values,
                "title": {"text": "<b>Confirmed Cases</b>"},
                "len": 0.8,
                "y": 1,
                "yanchor": "top"
            },
            "hovertemplate": df["text"],
            "name": "",
            "showlegend": False
        }
    )

    # 显示线的颜色和名称
    confirmed_cases_trace = go.Scatter(
        x=dates_list,
        y=confirmed_cases_totals[:i],
        mode="markers" if i == 1 else "lines",
        name="Total Confirmed Cases",
        line={"color": "Red"},
        hovertemplate="%{x}<br>Total confirmed cases: %{y:,}<extra></extra>"
    )

    # 显示线的颜色和名称
    deaths_trace = go.Scatter(
        x=dates_list,
        y=deaths_totals[:i],
        mode="markers" if i == 1 else "lines",
        name="Total Deaths",
        line={"color": "Black"},
        hovertemplate="%{x}<br>Total deaths: %{y:,}<extra></extra>"
    )

    # 第一帧显示的信息
    if i == 1:
        fig.add_trace(choro_trace, row=1, col=1)
        fig.add_traces([confirmed_cases_trace, deaths_trace], rows=[2, 2], cols=[1, 1])
    frames.append({"data": [choro_trace, confirmed_cases_trace, deaths_trace], "name": date})

    steps.append(
        {"args": [[date], frame_args(50)], "label": date, "method": "animate", }
    )

# 坐标轴和显示高宽等设置

fig.update_xaxes(range=[0, len(dates_list) - 1], visible=False)
fig.update_yaxes(range=[0, max(confirmed_cases_totals)])
fig.frames = frames
fig.layout.sliders[0].steps = steps
fig.layout.geo.domain = {"x": [0, 1], "y": [0.2, 1]}
fig.update_layout(
    height=650,
    legend={"x": 0.05, "y": 0.175, "yanchor": "top", "bgcolor": "rgba(0, 0, 0, 0)"})
fig.write_html("nCoV_tracker_chart1.html")

def chart_2():     # chart_2


    #字典映射
    renamed_columns_map = {
        "Country/Region": "country",
        "Province/State": "location",
        "Lat": "latitude",
        "Long": "longitude"
    }

    #获取数据
    confirmed_cases_df = (
        pd.read_csv(confirmed_cases_url)
            .rename(columns=renamed_columns_map)
            .rename(columns=reformat_dates)
            .fillna(method="bfill", axis=1)
    )
    deaths_df = (
        pd.read_csv(deaths_url)
            .rename(columns=renamed_columns_map)
            .rename(columns=reformat_dates)
            .fillna(method="bfill", axis=1)
    )

    display(confirmed_cases_df.head())
    display(deaths_df.head())


    fig = go.Figure()

    geo_data_cols = ["country", "location", "latitude", "longitude"]
    geo_data_df = confirmed_cases_df[geo_data_cols]
    dates_list = (
        confirmed_cases_df.filter(regex=r"(\d{2}/\d{2}/\d{4})", axis=1)
            .columns
            .to_list()
    )
    



    # 所有日期 国家病例数
    cases_by_date = {}
    for date in dates_list:

        confirmed_cases_day_df = (
            confirmed_cases_df.filter(like=date, axis=1)
                .rename(columns=lambda col: "confirmed_cases")
                .astype("uint32")
        )

        deaths_day_df = (
            deaths_df.filter(like=date, axis=1)
                .rename(columns=lambda col: "deaths")
                .astype("uint32")
        )

        cases_df = confirmed_cases_day_df.join(deaths_day_df)
        cases_df = geo_data_df.join(cases_df)
        cases_df = cases_df[cases_df["confirmed_cases"] > 0]

        cases_by_date[date] = cases_df


    cases_by_date[dates_list[-1]].head()

    # 列表存放日期帧 显示的所有国家地区的比例数
    fig.data = []
    # 循环 鼠标显示地区病例
    for date, df in cases_by_date.items():
        df["confirmed_cases_norm"] = np.log1p(df["confirmed_cases"])
        df["text"] = (
                date
                + "<br>"
                + df["country"]
                + "<br>"
                + df["location"]
                + "<br>Confirmed cases: "
                + df["confirmed_cases"].astype(str)
                + "<br>Deaths: "
                + df["deaths"].astype(str)
        )
        # 显示原点颜色面积
        fig.add_trace(
            go.Scattergeo(
                name="",
                lat=df["latitude"],
                lon=df["longitude"],
                visible=False,
                hovertemplate=df["text"],
                showlegend=False,
                marker={
                    "size": df["confirmed_cases_norm"] * 100,
                    "color": "red",
                    "opacity": 0.75,
                    "sizemode": "area",
                },
            )
        )

    # 添加描述文字模板
    annotation_text_template = "<b>Worldwide Totals</b>" \
                               "<br>{date}<br><br>" \
                               "Confirmed cases: {confirmed_cases:,d}<br>" \
                               "Deaths: {deaths:,d}<br>" \
                               "Mortality rate: {mortality_rate:.1%}"
    # 文字模板参数
    annotation_dict = {
        "x": 0.03,
        "y": 0.35,
        "width": 175,
        "height": 110,
        "showarrow": False,
        "text": "",
        "valign": "middle",
        "visible": False,
        "bordercolor": "black",
    }

    # 制作所有帧
    steps = []
    for i, data in enumerate(fig.data):
        step = {
            "method": "update",
            "args": [
                {"visible": [False] * len(fig.data)},
                {"annotations": [dict(annotation_dict) for _ in range(len(fig.data))]},
            ],
            "label": dates_list[i],
        }


        step["args"][0]["visible"][i] = True
        step["args"][1]["annotations"][i]["visible"] = True

        df = cases_by_date[dates_list[i]]
        confirmed_cases = df["confirmed_cases"].sum()
        deaths = df["deaths"].sum()
        mortality_rate = deaths / confirmed_cases
        step["args"][1]["annotations"][i]["text"] = annotation_text_template.format(
            date=dates_list[i],
            confirmed_cases=confirmed_cases,
            deaths=deaths,
            mortality_rate=mortality_rate,
        )

        steps.append(step)

    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "Date: "},
            "steps": steps,
            "len": 0.9,
            "x": 0.05,
        }
    ]

    first_annotation_dict = {**annotation_dict}
    # 更新模板文字  默认第一帧
    first_annotation_dict.update(
        {
            "visible": True,
            "text": annotation_text_template.format(
                date="10/01/2020", confirmed_cases=44, deaths=1, mortality_rate=0.0227
            ),
        }
    )
    # 标题
    fig.layout.title = {"text": "Covid-19 Global Case Tracker", "x": 0.5}
    fig.update_layout(
        height=650,
        margin={"t": 50, "b": 20, "l": 20, "r": 20},
        annotations=[go.layout.Annotation(**first_annotation_dict)],
        sliders=sliders,
    )
    fig.data[0].visible = True  # set the first data point visible
    fig

    # 高度
    fig.update_layout(height=1000)
    fig.write_html("nCoV_tracker_chart2.html")

chart_2()