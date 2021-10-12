'''集合的表示方法'''

'''用数组表示集合：数组存节点，节点包含两个数据：（1）节点数据（2）父亲节点的下标（具体见课件）

（每一个根节点（父亲节点）都是一个集合的表示）
'''


class Node_type:  # 定义集合数组中的“节点”类
    def __init__(self,data,parent):  # 包含：（1）自身存储的数据（2）自身父节点（所属集合）的下标
        self.data = data
        self.parent = parent


class Set:
    def __init__(self):
        self.set_ground = []

    def add_element(self,data,parent):  # 方法：向集合数组中添加元素或集合
        node = Node_type(data,parent)
        self.set_ground.append(node)

        if node.parent < 0:  # 默认集合的parent=-1，即含有一个元素
            print(f'集合{node.data}添加成功')  # 集合的根节点，也是该集合中的一个元素而已
        else:
            print(f'元素{node.data}添加成功')  # 若parent>0，意味着这是某个集合中的元素
            # （注意此处是-=）
            self.set_ground[parent].parent -= 1  # 集合中元素数量 计数（-2：2个  -n：n个）

    def find_set(self,find_data):  # 方法：寻找某个元素在数组中的下标 以及 其所属集合的下标
        tem = -100  # 临时变量
        # （这一步很浪费时间）
        for i in range(len(self.set_ground)):  # 从头寻找元素所在位置
            if self.set_ground[i].data == find_data:
                print(f'元素{find_data}下标为{i}')
                tem = i

        if tem == -100:  # 若tem还是-100，则代表没找到
            print(f'元素{find_data}不存在')
            return -1
        else:  # 找到了
            while self.set_ground[tem].parent >= 0:  # 循环：寻根溯源，找到最上层的父节点（所在的大集合）
                tem = self.set_ground[tem].parent
            print(f'元素{find_data}的所属集合{self.set_ground[tem].data}')
            return tem  # 返回所属集合的下标

    def union(self,data_1,data_2):  # 方法：集合的并运算（输入两个元素值，将这两个元素所在的两个集合，并运算）
        root_1 = self.find_set(data_1)
        root_2 = self.find_set(data_2)  # 首先找到两个元素的集合下标

        if root_1 == root_2:  # 若一样
            print(f'元素{data_1}和元素{data_2}所属同一个集合，无需要并运算')
        else:  # 否则
            # 我们要把较小的集合，并到较大的集合中去，这样可以避免树越来越高
            if self.set_ground[root_1].parent < self.set_ground[root_2].parent:  # 谁的parent更小，意味着谁的元素数量越多
                self.set_ground[root_1].parent += self.set_ground[root_2].parent  # 计数（注意此处是+=）
                self.set_ground[root_2].parent = root_1  # 所谓“并运算”，就是把一个集合变成另一个集合的儿子
                return
            else:
                self.set_ground[root_2].parent += self.set_ground[root_1].parent
                self.set_ground[root_1].parent = root_2
                return

    def display_element_number(self):  # 方法：显示数组中所有集合，以及包含元素数量
        for i in range(len(self.set_ground)):
            if self.set_ground[i].parent < 0:
                print(f'集合{self.set_ground[i].data}的元素数量为{-self.set_ground[i].parent}')


aa = Set()
aa.add_element(0,-1)
aa.add_element(1,-1)
aa.add_element(2,-1)
aa.add_element(10,0)
aa.add_element(101,1)
aa.add_element(7740,0)
aa.add_element(2002,2)
aa.add_element(802,2)
aa.display_element_number()
print('--------')
aa.find_set(0)
aa.find_set(2002)
aa.find_set(7740)
aa.find_set(101)
aa.find_set(3333333333)
print('--------')
aa.union(7740,7740)
aa.union(7740,2002)
aa.find_set(7740)
aa.display_element_number()
