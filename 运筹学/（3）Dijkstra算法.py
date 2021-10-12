"""
单源最短路问题——Dijkstra算法
"""
'''经典的“Dijkstra”算法'''
'''若存在负数的权值，则可能出现“负值圈（negative-cost cycle）”,所以以下不考虑该情况'''
'''有向'''
'''有明确起点、不存在负数权值'''

'''和运筹学笔记中的思路一样，一个一个顶点被收录，变成整体，慢慢向外继续寻找扩展'''
import numpy as np


class Edge:  # 定义“边”类
    def __init__(self,s,t,weight):
        self.s = s  # 出发点
        self.t = t  # 结束点
        self.weight = weight  # 权值


class Dijkstra:  # 定义“图”类
    def __init__(self,vertex_num):
        self.vertex_num = vertex_num  # 图中所含节点数量
        self.vertex_data = [None] * vertex_num  # 节点信息（有时候顶点是有意义的，要存一些东西）
        self.edge_num = 0  # 图中边数量

        self.graph = np.ones((vertex_num,vertex_num)) * float('inf')  # 初始化矩阵（全是距离无穷）
        for i in range(vertex_num):
            self.graph[i][i] = 0  # 自己到自己距离为0

        self.dist_vertex = [float('inf')] * vertex_num  # 存放从起点出发，到各点的最短距离（正无穷inf即为没访问过该点）
        self.path_vertex = [-1] * vertex_num            # 存放路径中各点的上一个点是什么（方便后续逆顺序找路径）
        self.collected_vertex = [False] * vertex_num    # 记录，每个点是否被收录进入整体部分

    def insert_edge(self,s,t,weight):  # 方法：插入两点之间的边
        edge = Edge(s,t,weight)

        self.graph[edge.s][edge.t] = edge.weight
        # 若是“无向图”，还需下一句（对称输入）
        # self.graph[edge.t][edge.s] = edge.weight

        self.edge_num += 1

    def build_graph(self,edge_list,data_list):  # 方法：建立完整的图
        # 输入边列表，节点信息列表
        for i in range(len(edge_list)):
            self.insert_edge(edge_list[i][0],edge_list[i][1],edge_list[i][2])

        for i in range(len(data_list)):
            self.vertex_data[i] = data_list[i]

    def algorithm(self,start):  # 方法：算法主体
        # 首先将起始点录入整体部分，初始化
        self.collected_vertex[start] = True
        # 初始化起始点出发
        for i in range(self.vertex_num):
            if self.graph[start][i] < float('inf'):
                self.dist_vertex[i] = self.graph[start][i]  # 初始化直接邻接点距离

                if self.graph[start][i] != 0:
                    self.path_vertex[i] = start             # 初始化直接邻接点路径
        # 循环：每次录入一个距离 起点 最短的节点（直到全部录入，停止循环）
        while True:
            tem_min_dist = float('inf')  # 临时最短距离
            tem_v = -1  # 临时节点

            for i in range(self.vertex_num):  # 循环：找到未收录的、且距离起点dist最小的节点
                if (self.collected_vertex[i] is False) and (self.dist_vertex[i] < tem_min_dist):
                    tem_min_dist = self.dist_vertex[i]
                    tem_v = i
            if tem_v == -1:  # 若没找到，说明全部处理完成（或其余点都是不连通的）
                break

            self.collected_vertex[tem_v] = True  # 将该节点录入整体部分

            for i in range(self.vertex_num):  # 循环：对该节点的每一个直接邻接点，且未收录的，判断距离长短并更新
                if (self.collected_vertex[i] is False) and (self.graph[tem_v][i] < float('inf')):
                    # 若收录后，节点长度更短，则更新dist
                    if self.dist_vertex[tem_v] + self.graph[tem_v][i] < self.dist_vertex[i]:
                        self.dist_vertex[i] = self.dist_vertex[tem_v] + self.graph[tem_v][i]
                        self.path_vertex[i] = tem_v

        self.display_route(start)  # 输出
        self.refresh()  # 刷新

    def refresh(self):  # 方法：重置节点状态（距离、路径、收录情况），方便下次使用
        for i in range(self.vertex_num):
            self.dist_vertex[i] = float('inf')
            self.path_vertex[i] = -1
            self.collected_vertex[i] = False

    def display_route(self,start):  # 方法：显示出完整结果
        print(f'起始点：{start}')
        for i in range(self.vertex_num):  # 对每个点的最短路径 进行一次
            if i == start:
                print('\n', f'{self.vertex_data[i]}:自身，无路径', f'距离为{self.dist_vertex[i]}')
                continue

            if self.path_vertex[i] == -1:
                print('\n', f'{self.vertex_data[i]}:不连通，无路径')
                continue

            tem = self.path_vertex[i]
            temporary_list = [i]
            print('\n', f'{self.vertex_data[i]}:', end='')
            while tem != -1:
                temporary_list.append(tem)
                tem = self.path_vertex[tem]
            while temporary_list:
                print(f'-> {temporary_list.pop()}', end='')
            print(f' 距离为{self.dist_vertex[i]}')


if __name__ == '__main__':
    Graph = Dijkstra(9)
    ve_list = [[0,1,6],[0,2,3],[0,3,1],
               [1,4,1],
               [2,1,2],[2,3,2],
               [3,5,10],
               [4,3,6],[4,5,4],[4,6,3],[4,7,6],
               [5,4,10],[5,6,2],
               [6,7,4],
               [8,4,2],[8,7,3]]
    v_data = ['点0','点1','点2','点3','点4','点5','点6','点7','点8']
    Graph.build_graph(ve_list,v_data)
    print('-----------')
    Graph.algorithm(0)
    print('-----------')
    Graph.algorithm(3)

'''
“Dijkstra”算法的拓展
（拓展一）同时考虑路径距离和总费用：首先优先考虑距离最短————其次，若有相同距离的路线，选择费用最低的
思路：
（1）原本的graph用于存放各边的权值，现在再增加一个图，用于存放各边的费用
（2）同时增加cost_vertex，记录起始点到各点的总费用（优先考虑距离前提下的，最小费用）（同理dist_vertex）
（3）先正常运用Dijkstra算法，在距离相同时，考虑费用最低

（拓展二）要求计算最短路径有多少条：
（1）加一个计数数组count[start] = 1
（2）如果找到了更短路径，则令count[i(新纳入的节点)] = count[v(上一个节点)] ，因为没有变嘛，所以直接赋值过来就行
（3）如果找到了等长路径，则令count[i(新纳入的节点)] += count[v(上一个节点)] ，因为每多一条等长的，相当于前面的路径数量就多出来一份

（拓展三）要求边数最少的最短路：
同理（拓展一）的距离-费用拓展，把所有的费用换成1，就相当于边数量了
'''
