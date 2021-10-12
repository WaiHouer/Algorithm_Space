'''冲突处理——平方探测法的实现'''
import math


def is_prime(n):  # 函数：判断是否为素数
    if n == 1:
        return False
    for i in range(2, int(math.sqrt(n)+1)):
        if n % i == 0:
            return False
    return True


def next_prime(number):  # 函数：寻找比number大一点点的素数且满足4k+3
    if is_prime(number) is True and (number-3)/4 == 0:
        return number  # 如果number就是,那么返回
    else:
        k = 0
        while True:
            if is_prime(4*k+3) is True and 4*k+3 > number:
                break
            else:
                k += 1
        return 4*k+3


class Hash_element:  # 哈希表每个元素包含两个内容，一个是元素值data（我们以字符串为例），一个是此处的状态（空、不空、已删除）
    def __init__(self,data,info):
        self.data = data
        self.info = info


class Hash_table:
    def __init__(self,table_size):
        self.table_size = next_prime(table_size)  # 不一定非要用table_size,找到他的下一个特殊素数
        self.hash_table = []
        for i in range(self.table_size):  # 创建初始哈希表，都是empty
            tem = Hash_element(None,'Empty')
            self.hash_table.append(tem)

    def h_key(self, char):  # 散列函数：字符串首字母取余
        key = ord(char[0]) % self.table_size
        return key

    def find_position(self,char):  # 函数：解决冲突的同时，确定最终存储位置
        conflict_num = 0  # 冲突次数
        current_position = self.h_key(char)  # 一开始计算得到的，位置
        new_position = current_position  # 初始化最终位置

        while self.hash_table[new_position].info != 'Empty' and self.hash_table[new_position].data != char:
            # 循环：当此处是空的 或 元素是将要存储的元素自己，停止循环
            conflict_num += 1  # 每次冲突次数+1

            if conflict_num % 2:  # 如果是奇数次冲突：+1、+4、+9、...
                new_position = int(current_position + ((conflict_num + 1) / 2) * ((conflict_num + 1) / 2))
                while new_position >= self.table_size:
                    new_position -= self.table_size  # 超范围了，减去上限，回来

            else:  # 如果是偶数次冲突：-1、-4、-9、...
                new_position = int(current_position - (conflict_num / 2) * (conflict_num / 2))
                while new_position < 0:
                    new_position += self.table_size

        return new_position  # 最终位置

    def insert(self,char):  # 函数：插入（添加）元素
        position = self.find_position(char)  # 找位置

        if self.hash_table[position].info != 'full':  # 如果该位置没有元素，插入
            self.hash_table[position].info = 'full'
            self.hash_table[position].data = char
            print(f'元素{char}插入成功')
        elif self.hash_table[position].data == char:  # 如果有元素，且相等
            print(f'已有该元素{char},无须再插入')
        else:
            print('该位置已满，无法插入')  # 如果有元素，但不相等

    def insert_list(self,char_list):  # 函数：插入（添加）元素列表
        for i in range(len(char_list)):
            self.insert(char_list[i])

    def delete(self,char):  # 删除元素
        position = self.find_position(char)
        if self.hash_table[position].info != 'full':
            print('哈希表中该位置没有元素，也就意味着没有该元素，无法删除')
        elif self.hash_table[position].data != char:
            print('哈希表中该位置有元素，但不是该元素，无法删除')
        else:
            self.hash_table[position].data = None
            self.hash_table[position].info = 'Deleted'
            # 注意：删除不是完全拿走，要标记这里删除过元素，这样不会影响其他元素的查找和定位


list_1 = ['cos','sin','tan','cot','clock']
aaa = Hash_table(len(list_1))
aaa.insert_list(list_1)

aaa.insert('cot')
aaa.insert('wow')
aaa.insert('char')
for ii in range(aaa.table_size):
    print(aaa.hash_table[ii].data,aaa.hash_table[ii].info)
