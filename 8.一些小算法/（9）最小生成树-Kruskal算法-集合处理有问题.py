'''最小生成树'''

'''Kruskal算法（将森林合并成树）：不断收录最小权值的边（且不构成回路），已知合成-合成-合成，最终变成一棵树'''
'''稠密图比较好，复杂度——O(|E|*log|E|)'''

'''注意：insert和build中的vertex_data都有改动'''
import numpy as np


class Edge_type:  # 定义“边”类
    def __init__(self, v_1, v_2, weight):
        self.vertex_1 = v_1
        self.vertex_2 = v_2
        self.weight = weight


class Matrix_Graph:  # 定义“邻接矩阵-6.图”类
    def __init__(self, vertex_number):
        self.vertex_number = vertex_number

        self.vertex_data = []
        for i in range(vertex_number):
            # 有时候顶点是有意义的，要存一些东西（这里是该节点所在集合，默认每个节点的所在集合是自己）
            self.vertex_data.append(i)

        self.edge_number = 0
        self.graph = np.ones((vertex_number, vertex_number)) * float('inf')  # 建立空矩阵（没有边）（用inf）
        for i in range(self.vertex_number):  # 保证自己到自己是0
            self.graph[i][i] = 0

        self.mst_set = []  # 最终的，最小生成树所含的边
        self.edge_set = []  # 图中的所有边

    def insert_edge(self, v_1, v_2, weight):  # 方法：插入两点之间的边
        temporary_edge = Edge_type(v_1, v_2, weight)

        self.edge_set.append(temporary_edge)  # 添加一条边（无向图也是添加一次就行）

        self.graph[temporary_edge.vertex_1][temporary_edge.vertex_2] = temporary_edge.weight
        # 若是无向图，还需下一句（对称输入）：
        self.graph[temporary_edge.vertex_2][temporary_edge.vertex_1] = temporary_edge.weight
        self.edge_number += 1  # 图中边的数量，计数

    def build_graph(self, edge_number, v_e_list):  # 方法：通过输入一定规则的数组，建立完整的图
        if edge_number != 0:  # 如果要插入的边的数量不为0
            for i in range(edge_number):  # 每次读一行，插入边
                self.insert_edge(v_e_list[i][0], v_e_list[i][1], v_e_list[i][2])

    def kruskal(self):  # 方法：最小生成树Kruskal算法
        # 首先按照权值对图中所有边进行排序（从大到小）
        edge_in_order_set = self.bubble_sort_optimize(self.edge_set)

        # 若最小生成树的边中数量<n-1，且图中还有边剩余，循环
        while (len(self.mst_set) < self.vertex_number - 1) and (len(edge_in_order_set) > 0):
            # 每次取出一条最小的边，并从图中删去（用pop）
            current_edge = edge_in_order_set.pop()
            # 若两边的顶点，不属于同一集合，则（即：如果这条边加进来，不构成回路）
            if self.vertex_data[current_edge.vertex_1] != self.vertex_data[current_edge.vertex_2]:
                self.mst_set.append(current_edge)  # 加进来这条边
                v_1 = current_edge.vertex_1
                v_2 = current_edge.vertex_2
                while self.vertex_data[v_1] != v_1:  # 找到最顶层的集合，合并
                    v_1 = self.vertex_data[v_1]
                self.vertex_data[v_2] = v_1

        if len(self.mst_set) < (self.vertex_number - 1):  # 退出循环后，如果最小生成树中边数不足n-1
            print('该图不连通，最小生成树不存在')
            self.refresh_visited()
        else:
            self.display_route()
            self.refresh_visited()

    def refresh_visited(self):  # 方法：重置节点状态（距离、路径、收录情况），方便下次使用
        for i in range(self.vertex_number):
            self.vertex_data[i] = i
        self.mst_set = []
        self.edge_set = []

    def display_route(self):
        print('保留的边有：')
        for i in range(len(self.mst_set)):
            print(self.mst_set[i].vertex_1, '-> ', end='')
            print(self.mst_set[i].vertex_2, '==', end='')
            print(self.mst_set[i].weight)

        kruskal_matrix = np.ones((self.vertex_number, self.vertex_number)) * float('inf')
        for i in range(self.vertex_number):  # 保证自己到自己是0
            kruskal_matrix[i][i] = 0
        for i in range(len(self.mst_set)):
            kruskal_matrix[self.mst_set[i].vertex_1][self.mst_set[i].vertex_2] = \
                self.graph[self.mst_set[i].vertex_1][self.mst_set[i].vertex_2]
            # 因为是无向图，所以有：
            kruskal_matrix[self.mst_set[i].vertex_2][self.mst_set[i].vertex_1] = \
                self.graph[self.mst_set[i].vertex_2][self.mst_set[i].vertex_1]

        print('最小生成树矩阵是：')
        print(kruskal_matrix)

    def bubble_sort_optimize(self, a_need_sort):  # 冒泡排序（改装版-反向，从大到小）
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


def main():
    Graph = Matrix_Graph(7)
    # ve_list = [[0, 1, 2],
    #            [0, 3, 1],
    #            [1, 3, 3],
    #            [1, 4, 10],
    #            [2, 0, 4],
    #            [2, 5, 5],
    #            [3, 2, 2],
    #            [3, 5, 8],
    #            [3, 6, 4],
    #            [3, 4, 2],
    #            [4, 6, 6],
    #            [6, 5, 1]]
    ve_list = [[0, 1, 5], [0, 2, 6], [1, 2, 1], [1, 3, 2], [2, 3, 7], [2, 4, 5], [3, 4, 3], [4, 5, 2], [3, 5, 4]]
    Graph.build_graph(len(ve_list), ve_list)
    print('-----------')
    print(Graph.graph)
    print('-----------')
    Graph.kruskal()


if __name__ == '__main__':
    main()

'''
集合问题：假如3和4归并，则3和4同属于集合3
        在进行2和3归并，则2和3同属于集合2，导致2和4分别属于2和3，这种间接关系导致，没有正确的归并到一起
'''
