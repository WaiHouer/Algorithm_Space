'''集合的简化表示'''

'''
在上一节中，find函数里的第一步，即从头开始寻找元素位置下标，这一步非常浪费时间

我们干脆直接用0--(N-1)的数字，一一映射N个元素，这N个连续的整数，即为下标
每一个下标代表一个元素，这样数组每个节点只存一个数据--就是父节点的值
'''
'''以下我们以最简单的为例，即：元素0，对应映射刚好为0；元素1，对应映射刚好为1......'''


class Set:
    def __init__(self,max_size):
        self.set_ground = [None]*max_size
        self.elements_size = max_size - self.set_ground.count(None)

    def add_element(self,data,parent):  # 方法：向集合数组中添加元素或集合
        self.set_ground[data] = parent

        if parent < 0:  # 默认集合的parent=-1，即含有一个元素
            print(f'集合{data}添加成功')  # 集合的根节点，也是该集合中的一个元素而已
        else:
            print(f'元素{data}添加成功')  # 若parent>0，意味着这是某个集合中的元素
            # （注意此处是-=）
            while self.set_ground[parent] >= 0:
                parent = self.set_ground[parent]
            self.set_ground[parent] -= 1  # 集合中元素数量 计数（-2：2个  -n：n个）

    def find_set(self,find_data):  # 方法：寻找某个元素所属集合的下标（元素值）
        temporary_index = find_data
        while self.set_ground[temporary_index] >= 0:
            temporary_index = self.set_ground[temporary_index]
        return temporary_index  # 返回集合下标（元素值）

    def union(self,data_1,data_2):
        set_index_1 = self.find_set(data_1)
        set_index_2 = self.find_set(data_2)
        if self.set_ground[set_index_1] < self.set_ground[set_index_2]:
            self.set_ground[set_index_1] += self.set_ground[set_index_2]
            self.set_ground[set_index_2] = set_index_1
            return
        else:
            self.set_ground[set_index_2] += self.set_ground[set_index_1]
            self.set_ground[set_index_1] = set_index_2

    def display_element_number(self):  # 方法：显示数组中所有集合，以及包含元素数量
        for i in range(len(self.set_ground)):
            if self.set_ground[i] is not None and self.set_ground[i] < 0:
                print(f'集合{i}元素个数为{-self.set_ground[i]}')


x = Set(100)
x.add_element(0,-1)
x.add_element(1,-1)
x.add_element(2,-1)
x.add_element(3,0)
x.add_element(4,0)
x.add_element(5,0)
x.add_element(6,4)
x.add_element(7,2)
x.add_element(8,7)
x.display_element_number()
print('--------')
x.union(1,8)
x.display_element_number()
