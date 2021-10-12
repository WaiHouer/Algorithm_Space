'''最大堆的建立'''

'''即：给定N个已知的元素值，按最大堆的要求存放在一个数组中'''

'''
思路一：通过insert操作，一个一个插入元素，直接形成最大堆（复杂度O(N*logN)）

思路二：（1）首先将N个数值按顺序存入，形成一个完全二叉树（随便什么顺序都行）
       （2）再调整各节点位置，以满足最大堆的有序要求
       （复杂度为线性的）
以下介绍思路二
'''


class Heap_node:
    def __init__(self,data,left_son=None,right_bro=None):
        self.data = data
        self.left_son = left_son
        self.right_bro = right_bro


class Max_Heap:
    def __init__(self,capacity):  # 初始化最大堆的容量、堆数组、标兵、元素个数
        self.capacity = capacity
        self.elements = [Heap_node(None)]*(capacity+1)
        self.elements[0] = Heap_node(999999)  # 第一个位置放“哨兵”，树的元素下标从1开始
        self.elements_size = 0  # 元素个数，不算哨兵

    def create_heap(self,need_to_heap):  # 方法：针对一组数据，建立最大堆
        # 首先按照顺序随便建立一个二叉树，并存到数组中
        for i in range(len(need_to_heap)):
            self.elements[i+1] = Heap_node(need_to_heap[i])
            self.elements_size += 1

        # 原理：从后往前，依次寻找带儿子的节点，每找到一次进行一次 类似“最大堆删除”的操作，从而达到整个二叉树成为最大堆的效果
        for j in range(self.elements_size-1,0,-1):  # 从最后向前一个一个倒数（不含哨兵），碰到有儿子的节点才进入while循环

            # 从此处开始，与最大堆的删除原理一致
            parent = j
            temporary_data = self.elements[parent].data

            while parent*2 <= self.elements_size:
                child = parent * 2
                if (child != self.elements_size) and (self.elements[child].data < self.elements[child+1].data):
                    child += 1  # 下标替换成右儿子
                if temporary_data > self.elements[child].data:
                    break
                else:
                    self.elements[parent] = self.elements[child]
                    parent = child
            self.elements[parent] = Heap_node(temporary_data)

    def display(self):  # 方法：显示出完全二叉树结构（后续可以修改成为 建立二叉树方法）
        for s in range(1,self.elements_size+1):
            print(f'节点：{self.elements[s].data}',end=' -')
            if s*2 <= self.elements_size:
                print(f'左指针：{self.elements[s*2].data}',end=' -')
            else:
                print(f'左指针：{None}', end=' -')
            if s*2+1 <= self.elements_size:
                print(f'右指针：{self.elements[s*2+1].data}')
            else:
                print(f'右指针：{None}')


aa = Max_Heap(10)
aa.create_heap([2,23,5,6,4,1,23,2,7,33])
aa.display()
