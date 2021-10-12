'''什么是图'''

'''
点、边组成的图形，表示“多对多”的关系
顶点——V：vertex
边——E：edge
边是顶点对：(v,w) ∈ E，其中 v,w ∈ V
有向边：<v,w>，从v指向w
不考虑重边、自回路
'''
'''
无向图：边是双向的
有向图：有方向的
网络：带权重的图
'''
'''对于一个无向图，可以用邻接矩阵表示，但是会浪费一半的空间（因为是上下三角形对称），怎样节省？

用一个大小是“N*(N+1)/2”的一维数组存储｛G00,G10,G11,G20,G21,G22,.....｝
则Gij在一维数组中对应的下标是：( i*(i+1)/2+j )

对于无向图，我们令Gij = 1，表示两点之间有连接；Gij = 0，表示两点直接没有连接
对于网络，我们令Gij = <vi,vj>的权值，表示两点之间有连接；没有边怎么表示？？？————后面遇到了再解决
'''

'''优点：
（1）直观，简单
（2）方便——检查任意一对顶点间是否存在边
（3）方便——查找任意顶点的所有邻接点
（4）方便——计算任意顶点的度（包括出度、入度-有向图的概念）
'''
'''缺点：
（1）浪费空间：对于稀疏图，有大量无效元素（稠密图还是很合算的）
（2）浪费时间：统计一共多少边
'''
'''所以：图的表示方法极多，根据具体问题，具体选择'''
import numpy as np


class Edge_type:  # 定义“边”类
    def __init__(self,v_1,v_2,weight):
        self.vertex_1 = v_1
        self.vertex_2 = v_2
        self.weight = weight


class Matrix_Graph:  # 定义“邻接矩阵-6.图”类
    def __init__(self,vertex_number):
        self.vertex_number = vertex_number
        self.vertex_data = [None]*vertex_number  # 有时候顶点是有意义的，要存一些东西
        self.edge_number = 0
        self.graph = np.zeros([vertex_number,vertex_number])  # 建立空矩阵（没有边）（可以用0，也可以用inf）

    def insert_edge(self,v_1,v_2,weight):  # 方法：插入两点之间的边
        temporary_edge = Edge_type(v_1,v_2,weight)
        self.graph[temporary_edge.vertex_1][temporary_edge.vertex_2] = temporary_edge.weight
        # 若是无向图，还需下一句（对称输入）：
        self.graph[temporary_edge.vertex_2][temporary_edge.vertex_1] = temporary_edge.weight
        self.edge_number += 1  # 图中边的数量，计数

    def build_graph(self,vertex_number,edge_number,v_e_list,v_data_list):  # 方法：通过输入一定规则的数组，建立完整的图
        if edge_number != 0:  # 如果要插入的边的数量不为0
            for i in range(edge_number):  # 每次读一行，插入边
                self.insert_edge(v_e_list[i][0],v_e_list[i][1],v_e_list[i][2])

        for j in range(vertex_number):  # 插入节点包含的信息
            self.vertex_data[j] = v_data_list[j]


ggg = Matrix_Graph(5)
ve_list = [[1,2,7],
           [2,4,56],
           [3,4,43],
           [0,3,12],
           [0,2,33],
           [1,4,99]]
v_data = ['点1','点2','点3','点4','点5']
ggg.build_graph(5,6,ve_list,v_data)
print(ggg.graph)
print(ggg.vertex_data)
print('边数：',ggg.edge_number)
print('---------')
ggg.insert_edge(0,1,111)
print(ggg.graph)
print('边数：',ggg.edge_number)
