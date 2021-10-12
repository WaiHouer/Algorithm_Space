'''邻接表 表示法'''

'''每一行的非0元素存成一个链表，每个非0点还是要存两遍，不算很省'''
'''
G[0] -> 1 -> 2.....
G[1] -> 42 -> 32....
....
'''

'''优点：
（1）方便——查找任意顶点的所有邻接点
（2）节约“稀疏图”的空间——稀疏图，这个方法才合算（N个头指针 + 2E个节点）
（3）方便——计算 无向图 的任意顶点的度
'''
'''缺点：
（1）不方便——计算 有向图 的出度和入度
（2）不方便——检查任意一对顶点间是否存在边
'''
'''所以：图的表示方法极多，根据具体问题，具体选择'''


class Graph_Vertex_Node:  # 定义“邻接表开头顶点”类
    def __init__(self,vertex_data=None,first_edge=None):  # 传入“顶点内容”和“边”类
        self.vertex_data = vertex_data  # 每个顶点可能存一些内容
        self.first_edge = first_edge  # 指向每个顶点的第一条边（指针）


class Graph_Edge:  # 定义“边”类
    def __init__(self,vertex,weight):
        self.nxt_vertex = vertex  # 邻接点下标（这条边的开头就是上面的“类”，结尾就是邻接点）
        self.weight = weight
        self.nxt = None  # 指向下一个节点的指针


class List_Graph:
    def __init__(self,vertex_number):
        self.vertex_number = vertex_number
        self.edge_number = 0
        self.list_graph = []
        for i in range(vertex_number):
            self.list_graph.append(Graph_Vertex_Node(f'点{i+1}'))  # 默认顶点内容就是“点n”

    def insert_edge(self,v_1,v_2,weight):  # 方法：插入新节点（总是从表头插入）
        temporary_edge = Graph_Edge(v_2,weight)
        temporary_edge.nxt = self.list_graph[v_1].first_edge
        self.list_graph[v_1].first_edge = temporary_edge

        # 若是“无向图”，还需要下面：（对称输入）
        temporary_edge = Graph_Edge(v_1, weight)
        temporary_edge.nxt = self.list_graph[v_2].first_edge
        self.list_graph[v_2].first_edge = temporary_edge

    def build_graph(self,vertex_number,edge_number,v_e_list,v_data_list):  # 方法：建立完整的图
        # 与（1）同理
        if edge_number != 0:
            for i in range(edge_number):
                self.insert_edge(v_e_list[i][0],v_e_list[i][1],v_e_list[i][2])

        for j in range(vertex_number):
            self.list_graph[j].vertex_data = v_data_list[j]

    def display_list(self):  # 方法：显示全部链结构
        for i in range(self.vertex_number):
            print('\n',self.list_graph[i].vertex_data,f': {i}',end='')
            current_edge = self.list_graph[i].first_edge
            while current_edge.nxt is not None:
                print(f' ->({current_edge.weight}) {current_edge.nxt_vertex}',end='')
                current_edge = current_edge.nxt


x = List_Graph(5)
for ji in range(5):
    print(x.list_graph[ji].vertex_data)

ve_list = [[1,2,7],
           [2,4,56],
           [3,4,43],
           [0,3,12],
           [0,2,33],
           [1,4,99]]
v_data = ['点1','点2','点3','点4','点5']
x.build_graph(5,6,ve_list,v_data)

x.display_list()
# for ji in range(5):
#     if x.list_graph[ji].first_edge is None:
#         print('None')
#     else:
#         print(ji,' -> ',x.list_graph[ji].first_edge.nxt_vertex,'权值：',x.list_graph[ji].first_edge.weight)
