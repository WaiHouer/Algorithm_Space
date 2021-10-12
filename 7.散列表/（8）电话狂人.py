'''电话狂人'''
'''在一堆电话通话记录中，找到出现次数最多的电话号码'''

'''首先观察手机号特征：
1 2 3           4 5 6 7        8 9 10 11
前三位：网络识别号 中间四位：地区编码 最后四位：随机用户号码
我们用后五位，构建散列函数
'''

'''类似于单词词频统计'''
import math


def is_prime(n):  # 函数：判断是否为素数
    if n == 1:
        return False
    for i in range(2, int(math.sqrt(n)+1)):
        if n % i == 0:
            return False
    return True


def next_prime(n):  # 函数：寻找比n大一点点的素数
    prime = n
    while True:
        if is_prime(prime) is True and prime > n:
            break
        else:
            prime += 1
    return prime


class Hash_node:  # “元素节点”类
    def __init__(self,data):
        self.data = data  # 用来存电话号码
        self.count = 0    # 用来对每个电话计数
        self.nxt = None


class Hash_table:  # 哈希表，我们以整数元素为例
    def __init__(self,table_size):
        self.table_size = next_prime(table_size)  # 素数
        self.hash_table = []
        for i in range(self.table_size):  # 初始化哈希表全是 头节点
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
            current_node.count += 1
        else:
            current_node.nxt = tem
            current_node.nxt.count += 1

    def add_number_list(self,number_list):
        for i in range(len(number_list)):
            for j in range(2):
                self.add_number(number_list[i][j])

    def find_number_count(self,number):  # 函数：查找某个电话号码的计数

        if_find_position = -1

        for i in range(self.table_size):
            current = self.hash_table[i]

            while current is not None and current.data != number:
                current = current.nxt
            if current is not None and current.data == number:
                if_find_position = i
                number_count = current.count
                print(f'电话{number}出现的次数为{number_count}')
                break
            else:
                continue

        if if_find_position == -1:
            print(f'没有元素{number}')

    def find_super_man(self):

        count = 0
        super_man = None

        for i in range(self.table_size):
            current = self.hash_table[i]

            while current is not None:
                if current.count > count:
                    count = current.count
                    super_man = current.data
                current = current.nxt

        if count == 0:
            print('列表中没有元素')
        else:
            print(f'电话狂人是{super_man},出现次数是{count}')


n_list = [[15045974282,13277224505],
          [18882337006,13277224505],
          [15045974282,18882337006],
          [18882337006,110],
          [110,18882337006]]

superman = Hash_table(2*len(n_list))
superman.add_number_list(n_list)
superman.find_number_count(110)
superman.find_super_man()
