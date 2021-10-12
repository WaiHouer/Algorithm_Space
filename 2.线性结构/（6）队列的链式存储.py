'''队列的链式存储实现，这种结构 单链表'''

'''
注意：
此时堆栈的front位置，应该记录在链式结构的head位置，用于删除节点元素
   堆栈的rear位置，应该记录在链式结构的末尾位置，用于添加节点元素
   
   ！！！front 不能放在队列末尾位置，因为删除末尾元素后，front无法找到上一个元素位置
   
这样方便出栈操作
若记录在末尾位置，则每次需要遍历才能出战
'''


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
            print(f'元素{data}入队列成功')
        else:
            current_node = self.front   # 若不是第一个元素，则遍历，在最后添加节点
            while current_node.nxt is not None:
                current_node = current_node.nxt
            current_node.nxt = node
            self.rear = node
            print(f'元素{data}入队列成功')

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
            print(f'元素{need_to_delete_data}出队列成功')
            return need_to_delete_data

    def length(self):  # 方法：记录队列长度（即，元素个数）
        current_node = self.front
        i = 0
        while current_node is not None:  # 遍历，累加计数
            current_node = current_node.nxt
            i += 1
        return i


q = list_queue()
q.add_queue(231)
q.add_queue(3)
q.add_queue(46)

for i in range(q.length()):
    q.delete_queue()

q.delete_queue()
