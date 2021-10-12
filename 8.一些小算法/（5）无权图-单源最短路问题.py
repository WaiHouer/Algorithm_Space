'''无权图--单源最短路问题'''

'''因为也是一层一层向外寻找，所以，以BFS为基础，进行改动即可'''
'''无权 = 权值均为1'''
'''有向'''
'''无权图的最短路问题：可以理解为经过顶点数量最少的那条路径（找出起始点到图中任意连通节点的最短路）'''

'''以下是要用到的队列类'''


class Note_type:
    def __init__(self,data):
        self.data = data
        self.nxt = None


class list_queue:
    def __init__(self):
        self.front = None  # 队列开头指针
        self.rear = None   # 队列末尾指针

    def is_empty(self):  # 方法：判断队列是否为空
        return self.front is None

    def add_queue(self, data):  # 方法：入队列（从末尾入队列）
        node = Note_type(data)
        if self.front is None:   # 如果是第一个元素，那么队列开头和尾部都指向这个节点
            self.front = node
            self.rear = node
            # print(f'元素{data}入队列成功')
        else:
            current_node = self.front   # 若不是第一个元素，则遍历，在最后添加节点
            while current_node.nxt is not None:
                current_node = current_node.nxt
            current_node.nxt = node
            self.rear = node
            # print(f'元素{data}入队列成功')

    def delete_queue(self):  # 方法：出队列（从开头出队列）
        if self.is_empty():
            print('队列为空，无法出队列')
        else:
            need_to_delete = self.front
            if self.front == self.rear:  # 如果只有一个节点
                self.front = None
                self.rear = None
            else:
                self.front = self.front.nxt
            need_to_delete_data = need_to_delete.data
            del need_to_delete
            # print(f'元素{need_to_delete_data}出队列成功')
            return need_to_delete_data


'''以上是要用到的队列类'''
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

        self.dist_vertex = [-1]*vertex_number  # 记录起点到各个点的最短距离（-1即为没访问过该点）
        self.path_vertex = [-1]*vertex_number  # 记录起点到各个点的 上一个点是什么（方便后续逆顺序找路径）

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

    def bfs_unweighted_shortest(self,start_vertex):  # 方法：广度优先搜索（仅限一个连通分量,即start_vertex所在的连通分量）
        queue = list_queue()

        self.dist_vertex[start_vertex] = 0  # 记录自己到自己为0
        queue.add_queue(start_vertex)  # 入队列

        while queue.is_empty() is False:
            tem = queue.delete_queue()
            for i in range(self.vertex_number):
                if self.graph[tem,i] > 0:
                    # 对于每一个与start_vertex节点相连接的点 有：
                    if self.dist_vertex[i] == -1:  # 如果没被访问过

                        # 出发点到该节点的距离 = 上一个点的距离 + 1
                        self.dist_vertex[i] = self.dist_vertex[tem] + 1
                        # 出发点到该节点的 上一个点是tem
                        self.path_vertex[i] = tem
                        queue.add_queue(i)
        self.display_route(start_vertex)  # 显示出来
        self.refresh_visited()  # 记得刷新，方便下次

    def refresh_visited(self):  # 方法：重置节点被访问状态，方便下次使用
        for i in range(self.vertex_number):
            self.dist_vertex[i] = -1
            self.path_vertex[i] = -1

    def display_route(self,start_vertex):  # 方法：显示出
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
    Graph = Matrix_Graph(8)
    ve_list = [[1, 2, 1],
               [2, 4, 1],
               [3, 4, 1],
               [0, 3, 1],
               [0, 2, 1],
               [1, 4, 1],
               [5, 6, 1],
               [5, 7, 1]]
    v_data = ['点0', '点1', '点2', '点3', '点4', '点5', '点6', '点7']
    Graph.build_graph(8, 6, ve_list, v_data)
    Graph.bfs_unweighted_shortest(0)
    print('-----------')
    Graph.bfs_unweighted_shortest(1)


if __name__ == '__main__':
    main()
