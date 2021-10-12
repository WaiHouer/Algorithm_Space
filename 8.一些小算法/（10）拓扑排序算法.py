'''拓扑排序'''


'''有一堆课程（节点），每门课程都有一些（或没有）前置课程，必须先学完前置课程，才可以学后面的课程

怎样排每学期的课程表合理呢？？
'''
'''每个节点代表一个特定的事件，节点之间的有向线段代表因果关系，形成的图：AOV（Activity On Vertex）网络'''
'''
拓扑序：如果图中从v到w有一条有向路径，则v一定排在w前面，满足此条件的顶点序列——拓扑序
拓扑排序：获得一个拓扑序的过程

AOV如果有合理的拓扑序，则必定是有向无环图（Directed Acyclic Graph , DAG）
'''
'''
思路：
（1）每次输出，入度为0的顶点
（2）将它们的邻接点的入度-1，继续循环
（3）其中，若大循环未完成就找不到入度=0的节点了，说明有回路
'''
import numpy as np


class Edge_type:  # 定义“边”类
    def __init__(self, v_1, v_2, weight):
        self.vertex_1 = v_1
        self.vertex_2 = v_2
        self.weight = weight


class Matrix_Graph:  # 定义“邻接矩阵-6.图”类
    def __init__(self, vertex_number):
        self.vertex_number = vertex_number
        self.vertex_data = [None]*vertex_number

        self.edge_number = 0
        self.graph = np.ones((vertex_number, vertex_number)) * float('inf')  # 建立空矩阵（没有边）（用inf）
        for i in range(self.vertex_number):  # 保证自己到自己是0
            self.graph[i][i] = 0

        self.in_degree = [0]*vertex_number  # 记录每个节点的 入度

    def insert_edge(self, v_1, v_2, weight):  # 方法：插入两点之间的边
        temporary_edge = Edge_type(v_1, v_2, weight)

        self.graph[temporary_edge.vertex_1][temporary_edge.vertex_2] = temporary_edge.weight
        # 若是无向图，还需下一句（对称输入）：
        # self.graph[temporary_edge.vertex_2][temporary_edge.vertex_1] = temporary_edge.weight
        self.in_degree[v_2] += 1  # 入度+1
        self.edge_number += 1  # 图中边的数量，计数

    def build_graph(self, edge_number, v_e_list, v_data_list):  # 方法：通过输入一定规则的数组，建立完整的图
        if edge_number != 0:  # 如果要插入的边的数量不为0
            for i in range(edge_number):  # 每次读一行，插入边
                self.insert_edge(v_e_list[i][0], v_e_list[i][1], v_e_list[i][2])
        for j in range(self.vertex_number):  # 插入节点包含的信息
            self.vertex_data[j] = v_data_list[j]

    def top_sort(self):
        out_queue = []  # 初始化输出队列（堆栈、列表、数组等等啥都可以）
        # 用队列最好，因为可以把先导课程全输出完成后，再看其他的，若是堆栈，可能就会一条路走到黑
        count = 0
        for i in range(self.vertex_number):
            if self.in_degree[i] == 0:
                out_queue.append(i)
        while out_queue:
            v = out_queue.pop(0)  # 队列pop
            print(self.vertex_data[v],'->')
            count += 1
            for i in range(self.vertex_number):
                if self.graph[v][i] < float('inf'):
                    self.in_degree[i] -= 1  # 先导课程输出完了，入度-1
                    if self.in_degree[i] == 0:
                        out_queue.append(i)
        if count != self.vertex_number:
            print('图中有回路')


def main():
    Graph = Matrix_Graph(15)
    ve_list = [[0,2,1],
               [1,2,1],
               [3,4,1],
               [4,5,1],
               [2,6,1],
               [7,8,1],
               [6,9,1],
               [8,9,1],
               [6,10,1],
               [8,10,1],
               [6,11,1],
               [1,12,1],
               [9,13,1],
               [5,14,1]]
    v_data = ['1：程序设计基础','2：离散数学','3：数据结构','4：微积分一','5：微积分二','6：线性代数','7：算法分析',
              '8：逻辑与计算机基础','9：计算机组成','10：操作系统','11：编译原理','12：数据库','13：计算理论',
              '14:计算机网络','15：数值分析']
    Graph.build_graph(len(ve_list),ve_list,v_data)
    print('-----------')
    print(Graph.graph)
    print('-----------')
    Graph.top_sort()


if __name__ == '__main__':
    main()
