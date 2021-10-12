'''多源最短路算法'''

'''方法一：直接将单源最短路算法调用N次——对稀疏图效果较好——O(|V|^3+|E|*|V|)
   方法二：Floyd算法——对稠密图效果较好——O(|V|^3)
'''

'''
思路：Dk[i][j] = 路径{i -> {l<k} -> j}的最小长度（即，经过k个顶点的最短路径）
所以——D0,D1,D2,...,D(V-1)给出了i到j真正的最短距离（递推）
最初的D(-1)就是正常的图矩阵（不直接相连用inf）

当D(k-1)已经完成，递推到D(k)时：若k不属于路径{i -> {l<k} -> j}，即不影响i到j的距离，则 Dk = D(k-1)
                            若k属于路径{i -> {l<k} -> j}，则该路径一定由两段最短路径组成 Dk[i][j] = D(k-1)[i][k] + D(k-1)[k][j]
可以看出，与单源最短路有些相似的思想
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
        self.graph = np.ones((vertex_number,vertex_number))*float('inf')  # 建立空矩阵（没有边）（用inf）
        for i in range(self.vertex_number):  # 保证自己到自己是0
            self.graph[i][i] = 0

        self.path = np.ones((self.vertex_number,self.vertex_number))*(-1)  # 初始化路径矩阵（每个元素代表从i到j的一个必经节点）

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

    def floyd(self):  # 方法：多源最短路算法（Floyd算法）
        # 首先初始化递推矩阵
        floyd_matrix = self.graph

        for k in range(self.vertex_number):  # 矩阵进行，k次递推（每次考虑k点是否在路径中）
            # 每次递推都，依次对每个节点进行判断和更新
            for i in range(self.vertex_number):
                for j in range(self.vertex_number):
                    # 若两段之和更短一些，就更新
                    if floyd_matrix[i][k] + floyd_matrix[k][j] < floyd_matrix[i][j]:
                        floyd_matrix[i][j] = floyd_matrix[i][k] + floyd_matrix[k][j]
                        self.path[i][j] = k  # 记录必经的节点，方便以此为界，分成前、后半段

        for i in range(self.vertex_number):
            for j in range(self.vertex_number):
                if floyd_matrix[i][j] < float('inf'):
                    print('\n',f'从点{i}到点{j}的最短距离是：{floyd_matrix[i][j]}，路径为：')
                    print(i,end='')
                    self.display_route(i,j)

    def display_route(self,i,j):  # 方法：显示出完整结果
        if self.path[i][j] == -1:
            print(' ->',j,end='')
            return
        k = int(self.path[i][j])
        self.display_route(i,k)  # 递归显示前半段
        self.display_route(k,j)  # 递归显示后半段


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
    Graph.floyd()


if __name__ == '__main__':
    main()
