'''最小堆的路径显示小程序'''

'''
给一组数据，通过程序整理成为最小堆，并且节点按顺序存入数组中
通过输入数组的下标，输出从该位置开始到根节点（下标1）的路径

'''


class Heap_node:
    def __init__(self,weight,left_son=None,right_bro=None):
        self.weight = weight  # 数据改为权值
        self.left_son = left_son
        self.right_bro = right_bro


class Min_Heap:
    def __init__(self,capacity):  # 初始化最大堆的容量、堆数组、标兵、元素个数
        self.capacity = capacity
        self.elements = [Heap_node(None)]*(capacity+1)
        self.elements[0] = Heap_node(-999999)  # 第一个位置放“哨兵”，树的元素下标从1开始
        self.elements_size = 0  # 元素个数，不算哨兵
        self.head = None  # 为了建树

    def create_heap(self,need_to_heap):  # 方法：针对一组数据，建立最小堆
        # 首先按照顺序随便建立一个二叉树，并存到数组中
        for i in range(len(need_to_heap)):
            self.elements[i+1] = Heap_node(need_to_heap[i])
            self.elements_size += 1

        # 原理：从后往前，依次寻找带儿子的节点，每找到一次进行一次 类似“最小堆删除”的操作，从而达到整个二叉树成为最小堆的效果
        for j in range(self.elements_size-1,0,-1):  # 从最后向前一个一个倒数（不含哨兵），碰到有儿子的节点才进入while循环

            # 从此处开始，与最小堆的删除原理一致
            parent = j
            temporary_data = self.elements[parent].weight

            while parent*2 <= self.elements_size:
                child = parent * 2
                if (child != self.elements_size) and (self.elements[child].weight > self.elements[child+1].weight):
                    child += 1  # 下标替换成右儿子
                if temporary_data < self.elements[child].weight:
                    break
                else:
                    self.elements[parent] = self.elements[child]
                    parent = child
            self.elements[parent] = Heap_node(temporary_data)

    def is_full(self):  # 方法：判断最大堆是否为满
        if self.elements_size < self.capacity:
            return 0
        else:
            return 1

    def is_empty(self):  # 方法：判断最大堆是否为空
        if self.elements_size == 0:
            return 1
        else:
            return 0

    def insert(self,heap_node):  # 方法：最小堆的插入（传入的是节点）
        if self.is_full():
            print('最小堆已满，无法继续插入元素')
            return

        i = self.elements_size + 1  # 首先默认添加到数组的最后面一位
        # i除以2向下取整，在数组中对应为第i个节点的根节点
        # 当根节点的值小于该节点时，则循环
        while self.elements[i//2].weight > heap_node.weight:  # 此时“哨兵”起了作用，碰到哨兵肯定就退出循环了，i便成了根节点
            self.elements[i] = self.elements[i//2]  # 调换根节点和该节点
            i = i//2
        self.elements[i] = heap_node  # 从最底层一路向上调换，直到合适位置退出循环，此时i为合适的位置下标
        print(f'元素{heap_node.weight}入最小堆成功')
        self.elements_size += 1  # 计数

    def delete_min(self):  # 方法：最小堆的删除
        # 原理：（1）将最后一个元素和第一个根节点元素调换（2）若正确，则不动；否则，一路向下调换直到正确位置
        if self.is_empty():
            print('最小堆已为空，无法继续出元素')
            return

        min_node = self.elements[1]  # 首先将根节点元素值记录下来，方便后续return
        temporary_data = self.elements[self.elements_size].weight  # 记录二叉树的最后一个元素值
        self.elements_size -= 1  # 元素数量-1

        parent = 1  # 首先将最后一个元素放到根节点的位置
        # 以下思路：已知左子树、右子树都是最大堆，我如何调整新的根节点？
        while parent*2 <= self.elements_size:  # 若根节有子树，则循环
            child = parent * 2  # 左儿子在数组中的下标

            # 以下操作：寻找左右儿子中最大的那一个，并让它和temporary比较

            # 如果（左儿子不是最后一个元素）并且（右儿子值 大于 左儿子值）
            if (child != self.elements_size) and (self.elements[child].weight > self.elements[child+1].weight):
                child += 1  # 下标替换成右儿子
            # 如果temporary比自己的左右儿子都大，那么就合乎最大堆的规则，退出循环
            if temporary_data < self.elements[child].weight:
                break
            else:  # 否则，将其替换上来，继续向下扫描
                self.elements[parent] = self.elements[child]
                parent = child

        self.elements[parent] = Heap_node(temporary_data)  # 从最上层一路向下调换，将寻找后的最终位置赋值
        print(f'元素{min_node.weight}出最小堆成功')
        return min_node

    def route(self,start_node_index):
        if start_node_index > self.elements_size:
            print('超出最小堆中元素个数')
            return

        print(f'下标为{start_node_index}的元素值为{self.elements[start_node_index].weight}')
        print('以此为起始点到根节点的路径是：',end='')
        while start_node_index >= 1:
            if start_node_index != 1:
                print(self.elements[start_node_index].weight,'-> ',end='')
            else:
                print(self.elements[start_node_index].weight)
            start_node_index = start_node_index // 2


def main():
    x = Min_Heap(100)
    x.insert(Heap_node(44))
    x.insert(Heap_node(35))
    x.insert(Heap_node(83))
    x.insert(Heap_node(12))
    x.insert(Heap_node(678))
    x.insert(Heap_node(378))
    x.insert(Heap_node(79))
    x.insert(Heap_node(57))
    x.insert(Heap_node(101))
    x.insert(Heap_node(60))
    for i in range(1,x.elements_size+1):
        print(x.elements[i].weight)
    print('---------')
    x.route(6)
    x.route(9)
    x.route(99)


if __name__ == '__main__':
    main()
