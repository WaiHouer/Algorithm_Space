'''电脑的连通集问题'''

'''问：有n台电脑，每两台之间可以选择连接一根网线，两台直接或间接连接的电脑认为是联通的
   如：1-2，2-4，4-7，则1和7也是联通的
'''
'''现在，给出一些连接情况，要求输出判断，指定的两台电脑是否连通的情况'''


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
        # 实现了“路径压缩”的查找运算
        temporary_index = find_data
        if self.set_ground[temporary_index] < 0:  # 此处没错误，不用管波浪线
            return temporary_index  # 若内容<0，则代表是集合，返回
        else:
            # 否则，代表该节点有父节点
            # 递归：每次都顺便把 该节点的 父节点的 5.集合 赋给它，这样一层一层下来
            # 整条查询路径上面的节点，都直接指向了最终的集合点，减少了大量树高！！
            # 虽然对于一次查询没什么作用，但是多次查询后，速度明显提升
            self.set_ground[temporary_index] = self.find_set(self.set_ground[temporary_index])
            # 该节点的集合  =  查询-它父节点所在的集合
            return self.set_ground[temporary_index]

    def union(self,data_1,data_2):  # 实现了“按秩归并”的并运算
        set_index_1 = self.find_set(data_1)
        set_index_2 = self.find_set(data_2)
        if set_index_1 == set_index_2:  # 若一样
            print(f'元素{data_1}和元素{data_2}所属同一个集合，无需要并运算')
        else:  # 否则
            if self.set_ground[set_index_1] < self.set_ground[set_index_2]:
                self.set_ground[set_index_1] += self.set_ground[set_index_2]
                self.set_ground[set_index_2] = set_index_1
                return
            else:
                self.set_ground[set_index_2] += self.set_ground[set_index_1]
                self.set_ground[set_index_1] = set_index_2

    def display_element_number(self):  # 方法：显示数组中所有集合，以及包含元素数量
        tem = 0
        for i in range(len(self.set_ground)):
            if self.set_ground[i] is not None and self.set_ground[i] < 0:  # 此处没错误，不用管标亮
                print(f'集合{i}元素个数为{-self.set_ground[i]}')
                tem += 1
        if tem == 0:
            print('网络中无任何连通集')
        elif tem == 1:
            print('网络已经整体连通')
        else:
            print(f'网络中包含{tem}个连通集')

    def check_same_set(self,data_1,data_2):
        set_index_1 = self.find_set(data_1)
        set_index_2 = self.find_set(data_2)
        if set_index_1 == set_index_2:
            print(f'电脑{data_1}和电脑{data_2}已连通')
        else:
            print(f'电脑{data_1}和电脑{data_2}未连通')


def main():
    # 初始化5台计算机
    computer_set = Set(5)
    computer_set.add_element(0, -1)
    computer_set.add_element(1, -1)
    computer_set.add_element(2, -1)
    computer_set.add_element(3, -1)
    computer_set.add_element(4, -1)

    # 连接指定的计算机
    computer_set.union(2,1)
    computer_set.union(3,4)
    computer_set.union(1,3)
    computer_set.union(1,3)

    # 检查某两台计算机是否连通
    computer_set.check_same_set(1,4)
    computer_set.check_same_set(2,3)
    computer_set.check_same_set(0,2)

    # 检查整个网络的连通集个数
    computer_set.display_element_number()


if __name__ == '__main__':
    main()
