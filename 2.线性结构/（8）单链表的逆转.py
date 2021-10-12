'''单链表的逆转'''

'''
有一个链表，存着1，2，3，4，5，6，7
编写程序，实现链表前N项逆转过来，即 N=4 —— 4，3，2，1，5，6，7
'''


class Node_type:        # 创建“节点”类
    def __init__(self, data):
        self.data = data   # “节点”包含：（1）存的数据（2）指向的下一个节点（仿指针）
        self.nxt = None    # 默认下一个指向 空


class single_list:      # 创建“链表”类，“链表”由无数个节点组成
    def __init__(self):  # 构造head指针
        self.head = None

    def is_empty(self):  # 方法：判断是否为空链表
        return self.head is None

    def append(self, data):  # 方法：在链表末尾加一个节点
        node = Node_type(data)
        if self.is_empty():  # 先判断是否空列表
            self.head = node  # 是空的，则表头给他
        else:
            current_node = self.head
            while current_node.nxt is not None:   # 不是空的，从头遍历至末尾，然后加上
                current_node = current_node.nxt
            current_node.nxt = node
        # print(f'链表末尾插入元素{data}成功')

    def travel(self):   # 方法：遍历链表并输出
        current_node = self.head
        i = 1
        print('\n','链表为：',end='')
        while current_node is not None:
            print(current_node.data,'->',end='')
            current_node = current_node.nxt
            i += 1

    def reverse(self, n):
        # 第一个元素认为是已经逆转好的
        new_head = self.head  # new_head代表逆转后的新头部，首先将第一个元素赋给它
        old_head = new_head.nxt  # old_head代表剩下未逆转部分的头部，从第二个元素开始
        for i in range(n-1):  # 要逆转前n个，刨去第一个，还剩(n-1)个要逆转
            temporary_node = old_head.nxt  # 临时节点，记录下一个old_head
            old_head.nxt = new_head  # 逆转指针

            new_head = old_head  # 依次将两个头部向后移一个节点
            old_head = temporary_node

        # 令new_head成为真正的头部，old_head接在后面
        self.head.nxt = old_head
        self.head = new_head


number_list = single_list()
for k in range(1,10):
    number_list.append(k)
number_list.travel()

N = 4
number_list.reverse(N)
number_list.travel()
