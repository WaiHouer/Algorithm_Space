'''六度空间-BFS'''

'''
六度空间（Six Degrees of Separation）-简称SDS：
每个人，想要和一个陌生人认识，中间通过不超过6个人就可以完成

设计一个程序：输出每个节点，在网络中，符合n度空间的节点个数
'''

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

    def bfs_sds(self,n,start_vertex):  # 方法：广度优先搜索（仅限一个连通分量,即start_vertex所在的连通分量）
        queue = list_queue()

        level = 0  # 广度搜索层数（n度空间要求n层）
        last_vertex = start_vertex  # last_vertex，用于记录每一层的最后一个节点元素（起始点，即为自身层的最后一个元素）
        tail = 0  # 初始化tail，记录下一层入队列的最后一个节点

        self.visited_vertex[start_vertex] = 1  # 记录已经访问过
        count_number = 1  # 计数，该节点周围符合n度空间的节点数（包含自己）
        queue.add_queue(start_vertex)  # 入队列

        while queue.is_empty() is False:
            tem = queue.delete_queue()   # 头一层依次出队列

            for i in range(self.vertex_number):  # 下一层依次入队列
                if self.graph[tem,i] > 0:
                    # 对于每一个与start_vertex节点相连接的点 有：
                    if self.visited_vertex[i] == 0:  # 如果没被访问过
                        self.visited_vertex[i] = 1
                        count_number += 1  # 计数
                        queue.add_queue(i)
                        tail = i  # tail，记录下一层入队列的最后一个节点（即，下一层最后一个节点）

            if tem == last_vertex:  # 如果头一层出队列的节点 = 头一层最后一个节点（这一层全出完了）
                level += 1  # 层数，计数
                last_vertex = tail  # 前进一层

            if level == n:
                break
        return count_number

    def bfs_sds_components(self,n):  # 方法：输出每一个节点，对应的n度空间节点
        for i in range(self.vertex_number):
            self.refresh_visited()  # 不要忘了每走一个节点，刷新一次访问状态
            if self.visited_vertex[i] == 0:
                count = self.bfs_sds(n,i)
                print(f'节点{i}的{n}度空间节点数为：{count}个')

        self.refresh_visited()

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
ggg.bfs_sds_components(2)
print('--------')
ggg.bfs_sds_components(1)

