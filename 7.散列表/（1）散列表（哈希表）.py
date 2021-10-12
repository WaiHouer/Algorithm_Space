'''
字符串的比较，是否可以转换为数字的比较，再进行处理？？
'''
'''腾讯QQ查找用户账号时，是怎么查找的？（动态的，随时有新用户插入，用户删除）'''

'''散列查找法：
（1）计算位置：构造散列函数，计算确定关键词存储位置
（2）解决冲突：应用某种策略，解决多个关键词位置相同的问题

————查找时间与问题规模无关！！
'''

'''
装填因子：假设散列表的大小为M，填入表中的元素个数为N，则称 a = N/M，为装填因子（即：占比率）
'''


class H_list_number:  # “哈希表”类——简单举例（存整数数字）
    def __init__(self,table_size):
        self.table_size = table_size
        self.h_list = [None] * table_size

    def h_key(self,number):  # 散列函数：取散列表大小的余数（简单举个例子而已）
        key = number % self.table_size
        return key

    def loading_factor(self):  # 装填因子
        loading_factor = (self.table_size - self.h_list.count(None))/self.table_size
        print(f'装填因子为{loading_factor}')

    def add_number(self,number):  # 添加单个元素
        location = self.h_key(number)
        if self.h_list[location] is None:
            self.h_list[location] = number
        else:
            print('此处已经有数字，冲突！')

    def add_number_list(self,number_list):  # 添加元素列表
        for i in range(len(number_list)):
            self.add_number(number_list[i])

    def find(self,number):  # 查找元素位置
        tem_location = self.h_key(number)
        if self.h_list[tem_location] is None:
            print(f'没有{number}该数字')
        elif self.h_list[tem_location] == number:
            print(f'{number}存放位置为{tem_location},找到！')
        else:
            print(f'{number}存放位置为{tem_location},已经存放别的数字，未找到！')


ohhh = H_list_number(17)
ohhh.add_number_list([18,23,11,20,2,7,27,30,42,15,34])
ohhh.loading_factor()
ohhh.find(20)
ohhh.find(53)
'''--------------------------------------------------------'''
import numpy as np


class H_list_char:  # “哈希表”类——简单举例（存字符串）
    def __init__(self,table_row,table_column):
        self.table_row = table_row
        self.table_column = table_column
        self.h_list = [[None]*table_column for i in range(table_row)]
        self.elements_number = 0

    def h_key(self,char):  # 散列函数：字符串首字母与“a”的差值
        key = ord(char[0]) - ord('a')
        return key

    def loading_factor(self):  # 装填因子
        loading_factor = self.elements_number/self.table_row*self.table_column
        print(f'装填因子为{loading_factor}')

    def add_char(self,char):
        location = self.h_key(char)
        if self.h_list[location][0] is None:
            self.h_list[location][0] = char
            self.elements_number += 1
        elif self.h_list[location][1] is None:
            self.h_list[location][1] = char
            self.elements_number += 1
        else:
            print(f'该位置已有两个元素，无法插入元素{char}')

    def add_char_list(self,char_list):
        for i in range(len(char_list)):
            self.add_char(char_list[i])

    def find(self,char):
        tem_location = self.h_key(char)
        if self.h_list[tem_location][0] is None:
            print('该位置没有字符串,未找到！')
        elif self.h_list[tem_location][0] == char or self.h_list[tem_location][1] == char:
            print(f'字符串{char}在哈希表的位置是{tem_location},找到！')
        else:
            print(f'字符串{char}在哈希表的位置是{tem_location},但已有其他字符串，未找到！')


heeee = H_list_char(26,2)
heeee.add_char_list(['acos','define','float','exp','char','atan','ceil','clck','ctime'])
heeee.add_char('dada')
heeee.loading_factor()
heeee.find('define')
heeee.find('clock')
heeee.find('dada')

