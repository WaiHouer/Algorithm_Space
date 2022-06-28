# _*_ coding:UTF-8 _*_
# 开发人员：10413
# 开发时间：2022-05-30 17:25
# 文件名： remote_control3.py
# 开发工具：PyCharm

import json
import requests
import pandas as pd
import numpy as np

AK = "ivj5ojidPH7GjWq609q32SpZ8lOGdi7y"  # 刘淇的AK


def getDistance(start, end):  # 1. 给定经纬度，可直接调用路线规划服务api求解距离
    """
         说明tactics选择为3：tactics，api默认值：0。可选值：
         0：常规路线，即多数用户常走的一条经验路线，满足大多数场景需求，是较推荐的一个策略
         1：不走高速
         2：躲避拥堵
         3：距离较短 √
         """

    url = "https://api.map.baidu.com/directionlite/v1/driving?origin={}&destination={}&tactics=3&ak={}".format(
        start,
        end,
        AK  # 自动调用，不用修改
    )
    res = requests.get(url)
    json_data = json.loads(res.text)

    if json_data["status"] == 0:
        return json_data["result"]["routes"][0]["distance"] / 1000  # 单位：km
    else:
        print(json_data["message"])
        return -1


# 导入数据
path = "./warehouse_location_update-湖北.xlsx"
data_base = pd.read_excel(path, sheet_name="warehouse0_location")
data_alternative = pd.read_excel(path, sheet_name="warehouse1_location")
data_netnode = pd.read_excel(path, sheet_name="warehouse2_location")
data_server = pd.read_excel(path, sheet_name="warehouse3_location")  # 新增网点地址，即服务中心,避免歧义，命名为server

# 所有基地仓到备选仓的实际距离
num_base = len(data_base)
num_alter = len(data_alternative)
num_netnode = len(data_netnode)
num_center = 5
num_lead = num_alter - num_center
num_server = len(data_server)

# split_data3 计算距离矩阵：服务中心到需求点距离
# 1. 初始化矩阵为0
shape = 20, num_netnode
server2demand_matrix3 = np.zeros(shape)

record_list3 = []  # 记录错误索引
for i in range(40, 60):  # 送装中心server
    for j in range(num_netnode):  # 需求点
        #  调api通过两点经纬度计算直线距离
        start_lat, start_lng = data_server["纬度"][i], data_server["经度"][i]
        end_lat, end_lng = data_netnode["纬度"][j], data_netnode["经度"][j]

        # 转换为字符串形式，调用api函数
        start = str(start_lat) + "," + str(start_lng)
        end = str(end_lat) + "," + str(end_lng)

        temp = getDistance(start, end)  # 调用api计算，server到需求点j的距离
        if temp == -1:  # 调用出错
            record_list3.append((i, j))  # 记录对应索引
            print("调用api失败! 索引：", (i, j))
        else:  # 成功调用
            server2demand_matrix3[i - 40][j] = temp

np.save('server2demand_matrix3.npy', server2demand_matrix3)
