'''有权图--单源最短路问题'''

'''经典的“Dijkstra”算法'''
'''若存在负数的权值，则可能出现“负值圈（negative-cost cycle）”,所以以下不考虑该情况'''
'''有向'''
'''有明确起点、不存在负数权值'''

'''和运筹学笔记中的思路一样，一个一个顶点被收录，变成整体，慢慢向外继续寻找扩展'''
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
        self.graph = np.ones((vertex_number,vertex_number))*float('inf')  # 建立空矩阵（没有边）（用inf）
        for i in range(self.vertex_number):  # 保证自己到自己是0
            self.graph[i][i] = 0

        self.dist_vertex = [float('inf')]*vertex_number  # 记录起点到各个点的最短距离（正无穷inf即为没访问过该点）
        self.path_vertex = [-1]*vertex_number  # 记录起点到各个点的 上一个点是什么（方便后续逆顺序找路径）
        self.collected_vertex = [False]*vertex_number  # 记录，每个点是否被收录进入整体部分

    def insert_edge(self, v_1, v_2, weight):  # 方法：插入两点之间的边
        temporary_edge = Edge_type(v_1, v_2, weight)
        self.graph[temporary_edge.vertex_1][temporary_edge.vertex_2] = temporary_edge.weight
        # 若是无向图，还需下一句（对称输入）：
        # self.graph[temporary_edge.vertex_2][temporary_edge.vertex_1] = temporary_edge.weight
        self.edge_number += 1  # 图中边的数量，计数

    def build_graph(self, vertex_number, edge_number, v_e_list, v_data_list):  # 方法：通过输入一定规则的数组，建立完整的图
        if edge_number != 0:  # 如果要插入的边的数量不为0
            for i in range(edge_number):  # 每次读一行，插入边
                self.insert_edge(v_e_list[i][0], v_e_list[i][1], v_e_list[i][2])

        for j in range(vertex_number):  # 插入节点包含的信息
            self.vertex_data[j] = v_data_list[j]

    def dijkstra(self,start_vertex):  # 方法：有向图-单源最短路算法
        # 首先将起始点录入整体部分，并进行初始化
        for i in range(self.vertex_number):  # 初始化起始点出发，最短距离
            if self.graph[start_vertex][i] < float('inf'):
                self.dist_vertex[i] = self.graph[start_vertex][i]  # 初始化直接邻接点距离
                if self.graph[start_vertex][i] != 0:
                    self.path_vertex[i] = start_vertex  # 初始化直接邻接点路径
        self.collected_vertex[start_vertex] = True  # 初始点录入整体部分
        # 循环：每次录入一个距离 起点 最短的节点
        while True:
            tem_min_dist = float('inf')  # 临时变量
            v = -1  # 临时节点
            for i in range(self.vertex_number):  # 循环：找到未收录的、且距离起点dist最小的节点
                if (self.collected_vertex[i] is False) and (self.dist_vertex[i] < tem_min_dist):
                    tem_min_dist = self.dist_vertex[i]
                    v = i
            if v == -1:  # 若没找到，说明全部处理完成（或其余点都是不连通的）
                break

            self.collected_vertex[v] = True  # 将该节点录入整体部分

            for i in range(self.vertex_number):  # 循环：对该节点的每一个邻接点，且未收录的，判断距离长短并更新
                if (self.graph[v][i] < float('inf')) and (self.collected_vertex[i] is False):
                    # 若收录后，节点长度更短，则更新dist
                    if self.dist_vertex[v] + self.graph[v][i] < self.dist_vertex[i]:
                        self.dist_vertex[i] = self.dist_vertex[v] + self.graph[v][i]
                        self.path_vertex[i] = v

        self.display_route(start_vertex)  # 输出
        self.refresh_visited()  # 刷新

    def refresh_visited(self):  # 方法：重置节点状态（距离、路径、收录情况），方便下次使用
        for i in range(self.vertex_number):
            self.dist_vertex[i] = float('inf')
            self.path_vertex[i] = -1
            self.collected_vertex[i] = False

    def display_route(self,start_vertex):  # 方法：显示出完整结果
        print(f'起始点：{start_vertex}')
        for i in range(self.vertex_number):  # 对每个点的最短路径 进行一次
            if i == start_vertex:
                print('\n',f'{self.vertex_data[i]}:自身，无路径',f'距离为{self.dist_vertex[i]}')
                continue

            if self.path_vertex[i] == -1:
                print('\n',f'{self.vertex_data[i]}:不连通，无路径')
                continue

            tem = self.path_vertex[i]
            temporary_list = [i]
            print('\n',f'{self.vertex_data[i]}:',end='')
            while tem != -1:
                temporary_list.append(tem)
                tem = self.path_vertex[tem]
            while temporary_list:
                print(f'-> {temporary_list.pop()}',end='')
            print(f' 距离为{self.dist_vertex[i]}')


def main():
    Graph = Matrix_Graph(9)
    ve_list = [[0, 1, 6], [0, 2, 3], [0, 3, 1],
               [1, 4, 1],
               [2, 1, 2], [2, 3, 2],
               [3, 5, 10],
               [4, 3, 6], [4, 5, 4], [4, 6, 3], [4, 7, 6],
               [5, 4, 10], [5, 6, 2],
               [6, 7, 4],
               [8, 4, 2], [8, 7, 3]]
    v_data = ['点0', '点1', '点2', '点3', '点4', '点5', '点6', '点7', '点8']
    Graph.build_graph(9,len(ve_list),ve_list, v_data)
    print('-----------')
    Graph.dijkstra(0)
    print('-----------')
    Graph.dijkstra(1)
    Graph.dijkstra(3)


if __name__ == '__main__':
    main()
