"""
最大流问题——Ford-Fulkerson算法（又称“标号法”）
"""
'''
通过不断寻找增广链，将网络中的流量填满，最后看发出多少即为最大流

增广链满足：（1）前向弧是非饱和弧（f < c）；（2）后向弧是非零弧（f > 0）

两种思路：BFS（广度优先）和DFS（深度优先）
'''
import numpy as np


class Edge:  # 定义“边”类
    def __init__(self,s,t,capacity,flow):
        self.s = s  # 出发点
        self.t = t  # 结束点
        self.capacity = capacity  # 容量
        self.flow = flow  # 流量


class Ford_Fulkerson:  # 定义“算法”类
    def __init__(self,vertex_num):
        self.vertex_num = vertex_num  # 图中所含节点数量
        self.vertex_data = [None] * vertex_num  # 节点信息（有时候顶点是有意义的，要存一些东西）
        self.edge_num = 0  # 图中边数量

        self.graph = np.full(shape=(vertex_num,vertex_num),fill_value=None)  # 初始化全是None的矩阵（后续存放节点）

        self.visited = [0] * vertex_num  # 标记每个点是否被检查过

    def insert_edge(self,s,t,capacity,flow):  # 方法：插入边
        # 输入起点，终点，容量，现有流量
        edge = Edge(s,t,capacity,flow)

        self.graph[s][t] = edge

        self.edge_num += 1

    def build_graph(self,edge_list,data_list):  # 方法：建立完整的图
        # 输入边列表，节点信息列表
        for i in range(len(edge_list)):
            self.insert_edge(edge_list[i][0],edge_list[i][1],edge_list[i][2],edge_list[i][3])

        for i in range(len(data_list)):
            self.vertex_data[i] = data_list[i]

    def algorithm(self,start,end,delta):  # 方法：算法主体：DFS思想
        # 输入起点，终点，增广链中流量的改变量（初始为一个极大的数）
        self.visited[start] = 1  # 首先，起点纳入被检查的集合

        if start == end:  # 如果起点等于终点，代表已经找到了一条增广链，返回此时的“改变量”
            return delta

        for i in range(self.vertex_num):  # 对起点周围的、未检查过的邻接点逐一排查
            if self.visited[i] == 0:
                if (self.graph[start][i] is not None) and (self.graph[start][i].flow < self.graph[start][i].capacity):
                    # 如果是前向弧，且有改进的余地
                    # 则，对当前的改变量进行更新（取小的）
                    delta_current = min(delta,self.graph[start][i].capacity - self.graph[start][i].flow)
                    # 递归，接着往下查找（DFS），必定会找到最深处，再进行其他支路，得到最终的改变量
                    delta_final = self.algorithm(i,end,delta_current)

                    if delta_final != 0:  # 改变量不等于0，说明此条路通，改变边上的流量
                        self.graph[start][i].flow += delta_final
                        return delta_final  # 同时，返回改变量（注意：一定要在这返回）
                    else:
                        continue  # 否则继续

                elif (self.graph[i][start] is not None) and (self.graph[i][start].flow > 0):
                    # 后向弧同理
                    delta_current = min(delta,self.graph[i][start].flow)
                    delta_final = self.algorithm(i,end,delta_current)

                    if delta_final != 0:
                        self.graph[i][start].flow -= delta_final
                        return delta_final
                    else:
                        continue

        return 0  # 否则，未能找到有效的增广链，改变量为0，此条链不动

    def max_flow(self,start,end):  # 方法：显示最大流量
        max_f = 0
        for i in range(self.vertex_num):
            if i != start:
                if self.graph[start][i] is not None:
                    max_f += self.graph[start][i].flow
                elif self.graph[i][start] is not None:
                    max_f += self.graph[i][start].flow
        print(f'该网络起点为{start},终点为{end},最大流为：{max_f}')


if __name__ == '__main__':
    Graph = Ford_Fulkerson(6)
    ve_list = [[0,1,5,1],[0,2,3,3],
               [1,3,2,2],
               [2,1,1,1],[2,4,4,3],
               [3,4,3,0],[3,5,2,1],
               [4,5,5,3]]
    v_data = ['点0','点1','点2','点3','点4','点5']
    Graph.build_graph(ve_list,v_data)
    Graph.algorithm(0,5,float('inf'))
    for p in range(6):
        for pp in range(6):
            if Graph.graph[p][pp]:
                print(p,'->',pp,'  ',Graph.graph[p][pp].capacity,Graph.graph[p][pp].flow)
    Graph.max_flow(0,5)
