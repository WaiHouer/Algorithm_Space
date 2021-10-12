'''冲突处理——分离链接法的实现'''


class Hash_node:  # “元素节点”类
    def __init__(self,data):
        self.data = data
        self.nxt = None


class Hash_table:  # 哈希表，我们以整数元素为例
    def __init__(self,table_size):
        self.table_size = table_size
        self.hash_table = []
        for i in range(table_size):  # 初始化哈希表全是 头节点
            self.hash_table.append(Hash_node('head'))

    def hash_key(self,number):
        key = number % self.table_size
        return key

    def add_number(self,number):
        position = self.hash_key(number)  # 计算出位置，在该位置插入链表中
        tem = Hash_node(number)

        current_node = self.hash_table[position]
        while current_node.nxt is not None and current_node.data != number:
            current_node = current_node.nxt
        if current_node.data == number:
            print(f'已有该元素{number}，无需插入')
        else:
            current_node.nxt = tem

    def add_number_list(self,number_list):
        for k in range(len(number_list)):
            self.add_number(number_list[k])

    def find_number(self,number):

        if_find_position = -1

        for i in range(self.table_size):
            current = self.hash_table[i]

            while current.nxt is not None and current.data != number:
                current = current.nxt
            if current.data == number:
                if_find_position = i
                print(f'元素{number}存储在位置{if_find_position}的链表上')
            else:
                continue
        if if_find_position == -1:
            print(f'没有元素{number}')

    def delete_number(self,number):

        if_find_position = 0

        for i in range(self.table_size):
            current = self.hash_table[i]

            while current is not None and current.data != number:
                tem = current  # 记录退出循环前的节点
                current = current.nxt
            if current is not None and current.data == number:
                if_find_position = i
                tem.nxt = current.nxt  # 删除节点（此处没错，不用管标亮）
                print(f'删除元素{number}成功')
            else:
                continue
        if if_find_position == 0:
            print(f'没有元素{number},无法删除')

    def display_all(self):
        for i in range(self.table_size):
            current = self.hash_table[i]
            print('\n',f'位置{i}: ',end='')
            while current.nxt is not None:
                print(current.nxt.data,'-> ',end='')
                current = current.nxt
        print('\n')


aaa = Hash_table(5)
aaa.add_number(41)
aaa.add_number(42)
aaa.add_number(42)
aaa.add_number_list([32,54,14,23,66,81,31,45])
aaa.display_all()

aaa.find_number(54)
aaa.find_number(99)

aaa.delete_number(81)
aaa.delete_number(99)
aaa.display_all()
