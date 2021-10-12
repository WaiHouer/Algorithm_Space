'''DFS'''

'''深度优先搜索（DFS）：Depth First Search'''

'''
类似于 “树的先序遍历”
（1）每次碰到一个新节点，先标记为已走过的点
（2）观察与该节点连接的四周节点
（3）只要找到一个没走过的，就接着走，直到找不到四周相连接的、未走过的节点
（4）此时，不能完事，原路返回
（5）每次返回一个，重复（2），（3）
（6）最终，直到返回到初始点，搜索结束

'''
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

        self.visited_vertex = [0]*vertex_number  # 记录哪些点被访问过

    def insert_edge(self, v_1, v_2, weight):  # 方法：插入两点之间的边
        temporary_edge = Edge_type(v_1, v_2, weight)
        self.graph[temporary_edge.vertex_1][temporary_edge.vertex_2] = temporary_edge.weight
        # 若是无向图，还需下一句（对称输入）：
        self.graph[temporary_edge.vertex_2][temporary_edge.vertex_1] = temporary_edge.weight
        self.edge_number += 1  # 图中边的数量，计数

    def build_graph(self, vertex_number, edge_number, v_e_list, v_data_list):  # 方法：通过输入一定规则的数组，建立完整的图
        if edge_number != 0:  # 如果要插入的边的数量不为0
            for i in range(edge_number):  # 每次读一行，插入边
                self.insert_edge(v_e_list[i][0], v_e_list[i][1], v_e_list[i][2])

        for j in range(vertex_number):  # 插入节点包含的信息
            self.vertex_data[j] = v_data_list[j]

    def dfs(self,start_vertex):  # 方法：深度优先搜索（仅限一个连通分量,即start_vertex所在的连通分量）
        print(self.vertex_data[start_vertex],' -> ',end='')  # 打印节点
        self.visited_vertex[start_vertex] = 1  # 记录已经访问过

        for i in range(self.vertex_number):
            if self.graph[start_vertex,i] > 0:
                # 对于每一个与start_vertex节点相连接的点 有：
                if self.visited_vertex[i] == 0:  # 如果没被访问过
                    self.dfs(i)  # 对它进行继续遍历

    def dfs_components(self):  # 方法：对于不连通的图，输出每一个连通分量的遍历结果
        components_number = 0
        for i in range(self.vertex_number):
            if self.visited_vertex[i] == 0:
                components_number += 1
                print('\n',f'第{components_number}个连通分量：')
                self.dfs(i)

    def refresh_visited(self):  # 方法：重置节点被访问状态，方便下次使用
        for i in range(self.vertex_number):
            self.visited_vertex[i] = 0


ggg = Matrix_Graph(8)
ve_list = [[1,2,7],
           [2,4,56],
           [3,4,43],
           [0,3,12],
           [0,2,33],
           [1,4,99]]
v_data = ['点1','点2','点3','点4','点5','点6','点7','点8']
ggg.build_graph(8,6,ve_list,v_data)
ggg.insert_edge(0,1,111)
ggg.insert_edge(5,6,66)  # 另一个连通分量的节点
ggg.insert_edge(5,7,80)
print(ggg.graph)
print('边数：',ggg.edge_number)
print('--------')
ggg.dfs(0)
ggg.refresh_visited()
print('--------')
ggg.dfs_components()
ggg.refresh_visited()
