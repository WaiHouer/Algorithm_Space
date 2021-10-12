'''链表结构存储'''


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
        print(f'链表末尾插入元素{data}成功')

    def travel(self):   # 方法：遍历链表并输出
        current_node = self.head
        i = 1
        while current_node is not None:
            print(f'链表第{i}个元素：',current_node.data)
            current_node = current_node.nxt
            i += 1

    def length(self):   # 方法：计算链表长度
        current_node = self.head
        i = 0
        while current_node is not None:   # 遍历，累加计数
            current_node = current_node.nxt
            i += 1
        return i

    def find_k_node(self, k):   # 方法：查找第k个位置的节点
        current_node = self.head
        i = 1
        while (current_node is not None) and (i < k):   # 从头遍历
            current_node = current_node.nxt
            i += 1
        if i == k:   # 判断为什么跳出循环，i=k即是找到了之后，正常跳出的
            return current_node   # 返回节点，注意：这是节点，想要看里面的值别忘了还要加上.data
        else:
            return None  # 链表找到头了也没找到，跳出循环

    def find_data_k(self, data):   # 方法：查找节点数据是data的位置
        current_node = self.head
        i = 1
        while (current_node is not None) and (current_node.data != data):
            current_node = current_node.nxt
            i += 1
        if current_node is not None:  # 遍历规则同上
            return i
        else:
            return None

    def insert_k_data(self, k, data):  # 方法：在第k个位置插入一个数据是data的节点
        node = Node_type(data)
        if k == 1:              # 判断是否为表头
            node.nxt = self.head
            self.head = node
            print(f'插入第{k}个节点成功')
        else:
            node_before = self.find_k_node(k-1)
            if node_before is None:        # 判断位置是否有效
                print(f'想要插入的节点{k}，其位置不合法')
            else:
                node.nxt = node_before.nxt
                node_before.nxt = node
                print(f'插入第{k}个节点成功')

    def delete_k_node(self, k):   # 方法：删除第k个位置上的节点
        if k == 1:        # 判断是否为表头
            self.head = self.head.nxt
            print(f'删除第{k}个节点成功')
        else:
            node_before = self.find_k_node(k-1)
            if node_before is None:    # 判断位置是否有效
                print(f'想要删除的节点{k}，其位置不合法')
            else:
                node_now = node_before.nxt
                node_after = node_now.nxt
                node_before.nxt = node_after
                del node_now  # 记得释放（其实我也不清楚有没有用）
                print(f'删除第{k}个节点成功')



x = single_list()
x.append(12)
x.append(339)
x.append(1028)
x.append(108)
x.append(38)
x.append(86)
x.append(10)
x.append([1,2,3])

x.insert_k_data(1,999)
x.insert_k_data(4,22)
x.insert_k_data(12,3)

# x.travel()

print('链表长度为：', x.length())
print('第0个节点是：', x.find_k_node(0))
print('第5个元素是：', x.find_k_node(5).data)
print('38在链表中的位置是：', x.find_data_k(38))
print('[1,2,3]在链表中的位置是：', x.find_data_k([1,2,3]))

x.insert_k_data(11,[1,1,1,1])
x.delete_k_node(14)
x.delete_k_node(5)
x.travel()