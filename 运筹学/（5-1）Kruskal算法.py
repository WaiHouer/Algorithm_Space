"""
解决寻找最小生成树问题（Minimum spanning tree-MST）—— Kruskal算法
"""
import numpy as np


def bubble_sort(a_need_sort):  # 冒泡排序（改装版-反向，从大到小）
    for i in range(1, len(a_need_sort)):
        tag = False  # 设立标记，记录这一趟循环是否发生了交换动作
        for j in range(0, len(a_need_sort) - i):
            if a_need_sort[j].weight < a_need_sort[j + 1].weight:
                temporary = a_need_sort[j]
                a_need_sort[j] = a_need_sort[j + 1]
                a_need_sort[j + 1] = temporary
                tag = True  # 发生了交换动作，代表这一趟没有浪费
        if tag is False:  # 每一趟判断一次，若没发生交换动作，就直接退出循环
            break
    return a_need_sort


class Edge:  # 定义“边”类
    def __init__(self,s,t,weight):
        self.s = s  # 起点
        self.t = t  # 终点
        self.weight = weight  # 权值


class Kruskal:  # 定义“Kruskal”类
    def __init__(self,vertex_num):
        self.vertex_num = vertex_num  # 顶点数

        self.vertex_data = []  # 存放顶点信息
        for i in range(vertex_num):
            # 有时候顶点是有意义的，要存一些东西（这里是该节点所在集合，初始每个节点的所在集合只有自己）
            self.vertex_data.append([i])

        self.graph = np.ones((vertex_num,vertex_num)) * float('inf')  # 矩阵
        for i in range(vertex_num):
            self.graph[i][i] = 0

        self.edge_mst_set = []  # 已经纳入最小生成树的边集合

        self.all_edge_set = []  # 图中所包含的所有边集合

    def insert_edge(self,s,t,weight):  # 方法：插入边
        edge = Edge(s,t,weight)

        self.all_edge_set.append(edge)  # 进入集合

        self.graph[s][t] = weight
        self.graph[t][s] = weight  # 无向图，对称

    def build_graph(self,edge_list):  # 方法：建立完整的图
        for i in range(len(edge_list)):
            self.insert_edge(edge_list[i][0],edge_list[i][1],edge_list[i][2])

    def algorithm(self):  # 方法：算法主体
        self.all_edge_set = bubble_sort(self.all_edge_set)  # 首先对所有边，按照权值大小，从大到小排序

        while (len(self.edge_mst_set) < self.vertex_num - 1) and (self.all_edge_set != []):
            # 当 “边数未达到 顶点数-1” 和 “还有边未收纳入树”

            edge = self.all_edge_set.pop()  # 每次取一个最小的边

            # 若两边的顶点，不属于同一集合，则（即：如果这条边加进来，不构成回路）
            if edge.t not in self.vertex_data[edge.s] and edge.s not in self.vertex_data[edge.t]:
                self.edge_mst_set.append(edge)  # 加进来这条边

                tem = self.vertex_data[edge.s]
                self.vertex_data[edge.s] += self.vertex_data[edge.t]  # 两个顶点的集合互相加一下
                self.vertex_data[edge.t] += tem

        # 最后退出循环，判断一下终止条件是啥，边数未达到但是没有边了，则是不存在树
        if len(self.edge_mst_set) < self.vertex_num - 1:
            print(len(self.edge_mst_set))
            print('该图不连通，最小生成树不存在')
        else:
            self.display_route()

    def display_route(self):  # 方法：显示最小生成树
        print('保留的边有：')
        for i in self.edge_mst_set:
            print(i.s, '-> ', end='')
            print(i.t, '==', end='')
            print(i.weight)

        kruskal_matrix = np.ones((self.vertex_num, self.vertex_num)) * float('inf')
        for i in range(self.vertex_num):  # 保证自己到自己是0
            kruskal_matrix[i][i] = 0
        for i in self.edge_mst_set:
            kruskal_matrix[i.s][i.t] = self.graph[i.s][i.t]
            # 因为是无向图，所以有：
            kruskal_matrix[i.t][i.s] = self.graph[i.t][i.s]

        print('最小生成树矩阵是：')
        print(kruskal_matrix)


if __name__ == '__main__':
    Graph = Kruskal(6)
    ve_list = [[0,1,5],[0,2,6],[1,2,1],[1,3,2],[2,3,7],[2,4,5],[3,4,3],[4,5,2],[3,5,4]]
    Graph.build_graph(ve_list)
    print('-----------')
    print(Graph.graph)
    print('-----------')
    Graph.algorithm()
