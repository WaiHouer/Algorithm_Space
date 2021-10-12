'''最小生成树'''

'''什么是“最小生成树”（Minimum Spanning Tree）？（有权无向图）
（1）是一棵树——无回路
           ——有n个顶点的话，必有(n-1)个边
（2）是生成树——包含所有顶点
           ——(n-1)个边都在图中有对应
（3）边的权值之和最小
'''

'''贪心算法：
（1）什么是“贪”：每一步都要最好的（贪心）
（2）什么是“好”：总要权值最小的边
（3）需要约束：（1——只能用图里有的边
             （2——只能刚好有(n-1)个边
             （3——不能有回路
'''
'''Prim算法（让小树长大）：运筹学中的避圈法（极其像Dijkstra算法——不断收录进整体，不断向外扩展找最短路）'''
'''稀疏图比较好，复杂度——O(|V|^2)'''
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

        self.dist_vertex = [float('inf')]*vertex_number  # 记录整体收录树，到各个点的最短距离（正无穷inf即为没访问过该点）
        self.collected_vertex = [False]*vertex_number  # 记录，每个点是否被收录进入整体部分

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

    def prim(self,start_vertex):  # 方法：最小生成树Prim算法（起点随便选）
        # 初始化父节点列表，方便追踪那两个节点之间的边被保留了下来
        parent = [-1]*self.vertex_number

        # 首先将起始点录入整体部分，并进行初始化
        for i in range(self.vertex_number):  # 初始化起始点出发，最短距离
            if self.graph[start_vertex][i] < float('inf'):
                self.dist_vertex[i] = self.graph[start_vertex][i]  # 初始化直接邻接点距离
                parent[i] = start_vertex  # 初始化起始点周围的父节点
        self.collected_vertex[start_vertex] = True  # 初始点录入整体部分
        parent[start_vertex] = -1  # 默认起始点的父节点为-1

        # 循环：每次录入一个距离 整体部分（3.树） 最短的节点
        while True:
            tem_min_dist = float('inf')  # 临时变量
            v = -1  # 临时节点
            # 最小堆，更好一些
            for i in range(self.vertex_number):  # 循环：找到未收录的、且距离整体部分dist最小的节点
                if (self.collected_vertex[i] is False) and (self.dist_vertex[i] < tem_min_dist):
                    tem_min_dist = self.dist_vertex[i]
                    v = i
            if v == -1:  # 若没找到，说明全部处理完成（或其余点都是不连通的）
                break

            self.collected_vertex[v] = True  # 将该节点录入整体部分
            self.dist_vertex[v] = 0  # 收录进入整体后，与整体的距离当然默认是0

            for i in range(self.vertex_number):  # 循环：对该节点的每一个邻接点，且未收录的，判断距离长短并更新
                if (self.graph[v,i] < float('inf')) and (self.collected_vertex[i] is False):
                    # 若收录后，有一条直接的边 < 原本的dist，更新
                    if self.graph[v][i] < self.dist_vertex[i]:
                        self.dist_vertex[i] = self.graph[v][i]
                        parent[i] = v  # 保留这个边，记录父节点

        if self.collected_vertex.count(True) < self.vertex_number:  # 如果不是所有点都被收录进来，则图可能不连通
            print('该图不连通，无最小生成树')
        else:
            self.display_mst(parent)

        self.refresh_visited()  # 刷新

    def refresh_visited(self):  # 方法：重置节点状态（距离、路径、收录情况），方便下次使用
        for i in range(self.vertex_number):
            self.dist_vertex[i] = float('inf')
            self.collected_vertex[i] = False

    def display_mst(self,parent):
        print('通过Prim算法，得到的最小生成树是:')
        # 初始化最小生成树矩阵，方便最后输出
        prim_matrix = np.ones((self.vertex_number, self.vertex_number)) * float('inf')
        for i in range(self.vertex_number):  # 保证自己到自己是0
            prim_matrix[i][i] = 0

        for v in range(self.vertex_number):
            v_parent = parent[v]
            if v_parent != -1:
                prim_matrix[v_parent][v] = self.graph[v_parent][v]  # 将保留下来的边依次填入矩阵中
                # 因为是无向图：
                prim_matrix[v][v_parent] = self.graph[v][v_parent]

        print(prim_matrix)


def main():
    Graph = Matrix_Graph(7)
    ve_list = [[0,1,2],
               [0,3,1],
               [1,3,3],
               [1,4,10],
               [2,0,4],
               [2,5,5],
               [3,2,2],
               [3,5,8],
               [3,6,4],
               [3,4,2],
               [4,6,6],
               [6,5,1]]
    v_data = ['点0', '点1', '点2', '点3', '点4', '点5', '点6', '点7']
    Graph.build_graph(7, len(ve_list), ve_list, v_data)
    print('-----------')
    Graph.prim(0)
    print('-----------')


if __name__ == '__main__':
    main()
