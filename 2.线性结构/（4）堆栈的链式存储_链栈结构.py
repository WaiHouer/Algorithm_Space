'''堆栈的链式存储实现，这种结构 单链表 叫做 “链栈” '''

'''
注意：此时堆栈的top位置，应该记录在链式结构的head位置
     这样方便出栈操作
     若记录在末尾位置，则每次需要遍历才能出战
'''


class Node_type:     # 定义“节点”类
    def __init__(self,data):
        self.data = data
        self.nxt = None


class list_stack:    # 定义“链栈”类
    def __init__(self):  # 初始化head
        self.head = None

    def is_empty(self):  # 方法：判断是否为空
        return self.head is None

    def length(self):   # 方法：计算链栈长度
        current_node = self.head
        i = 0
        while current_node is not None:   # 遍历，累加计数
            current_node = current_node.nxt
            i += 1
        return i

    def push(self,data):  # 方法：入栈（注意，每次入栈都相当于在队伍head插入一个节点）
        node = Node_type(data)
        node.nxt = self.head
        self.head = node
        print(f'元素{data}入栈成功')

    def pop(self):  # 方法：出栈（注意，每次出栈都相当于在队伍head输出并删除一个节点）
        if self.is_empty():
            print('堆栈为空，不可出栈')
        else:
            need_to_pop = self.head  # 中间变量记录出栈前的head节点
            self.head = need_to_pop.nxt
            need_to_pop_data = need_to_pop.data  # 交换后，记录出栈节点的数据
            del need_to_pop  # 删除该节点
            print(f'元素{need_to_pop_data}出栈成功')
            return need_to_pop_data  # 返回数据


x = list_stack()
x.push(20)
x.push(21)
x.push(58)
x.push(41)
for i in range(x.length()):
    x.pop()