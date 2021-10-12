'''以下是 堆排序 中要用到的最小堆类'''


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

    def insert(self,heap_node):  # 方法：最小堆的插入
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
        # print(f'元素{heap_node.weight}入最小堆成功')
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
        # print(f'元素{min_node.weight}出最小堆成功')
        return min_node


'''以下是对一些排序算法的小总结'''

'''如果遇到相等的值不进行交换，那这种排序方式是稳定的排序方式'''
'''稳定性：任意两个相等的数据，排序前后的相对位置不发生变化'''


'''
任何以“交换相邻两元素来排序的算法，其平均时间复杂度的下界是 Ω(n^2)”
意味着：想要提高效率，必须（1）每次消去不止一个逆序对（2）每次交换相隔较远的2个元素
'''

'''
（1）冒泡排序 —— 复杂度 O(n^2) —— 稳定 —— （交换次数 = 逆序对的个数）

——基本思路：每两个数字之间进行比较，小的放左边，大的放右边
          这样每跑一趟最大的数字就会在最右边“冒出来”
          所以每循环一次，就排除掉最后一个不参与计算（因为他是最大的）
          最后从左到右，从大到小排序完成
——过程：1和2比较，2和3比较......n-1和n比较   （第1趟，n-1次比较）
       1和2比较，2和3比较......n-2和n-1比较 （第2趟，n-2次比较）
       .....
       完成
——存在问题：若其中某次已经排序完成，剩下的循环排序过程只是平白浪费时间
'''


def bubble_sort(a_need_sort):  # 为了清晰表达，我在循环条件里加上了0
    for i in range(1, len(a_need_sort)):  # 总共进行(n-1)趟
        for j in range(0, len(a_need_sort)-i):  # 每趟进行(n-i)次比较，且下标从0开始
            if a_need_sort[j] > a_need_sort[j+1]:
                temporary = a_need_sort[j]
                a_need_sort[j] = a_need_sort[j+1]
                a_need_sort[j+1] = temporary
    return a_need_sort


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('冒泡排序：')
print(bubble_sort(a))
print('---------------')


def bubble_sort_optimize(a_need_sort):  # 对存在问题进行优化
    for i in range(1, len(a_need_sort)):
        tag = False   # 设立标记，记录这一趟循环是否发生了交换动作
        for j in range(0, len(a_need_sort)-i):
            if a_need_sort[j] > a_need_sort[j+1]:
                temporary = a_need_sort[j]
                a_need_sort[j] = a_need_sort[j+1]
                a_need_sort[j+1] = temporary
                tag = True  # 发生了交换动作，代表这一趟没有浪费
        if tag is False:  # 每一趟判断一次，若没发生交换动作，就直接退出循环
            break
    return a_need_sort


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('冒泡排序（改进）：')
print(bubble_sort(a))
print('---------------')


'''
（2）选择排序 —— 复杂度 O(n^2)

——基本思路：一次次进行遍历，每遍历一次选择出最小的元素
          与参与遍历的第一个元素交换
          直到所有元素选择完毕
——过程：1和min比较，2和min比较.......n和min比较，交换1和min  （第1趟，比较n次）
       2和min比较，3和min比较.......n和min比较，交换2和min  （第2趟，比较n-1次）
       .....
       完成
'''


def selection_sort(a_need_sort):  # 为了清晰表达，我在循环条件里加上了0
    for i in range(0, len(a_need_sort)-1):  # 总共(n-1)趟
        min_number_j = i  # 首先假设头一个元素最小，记录下标
        tag = False  # 交换标记
        for j in range(i, len(a_need_sort)):  # 每趟比较(n-i)次
            if a_need_sort[j] < a_need_sort[min_number_j]:
                min_number_j = j  # 更新最小元素下标
                tag = True
        if tag is True:  # 若发生了交换动作，则进行元素交换，否则不用动以防浪费
            temporary = a_need_sort[i]
            a_need_sort[i] = a_need_sort[min_number_j]
            a_need_sort[min_number_j] = temporary
    return a_need_sort


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('选择排序：')
print(selection_sort(a))
print('---------------')


'''
（3）插入排序 —— 复杂度 O(n^2) —— 稳定 —— （交换次数 = 逆序对的个数）

——基本思路：类似于抽扑克牌，每进行到一个数字，就从头比较，选择正确位置插入
——过程：2和1比较，选择正确位置插入
       3和2比较，（若正确则插入），2和1比较，（若正确则插入）
       4和3比较，（若正确则插入），3和2比较，（若正确则插入），2和1比较，（若正确则插入）
       .....
       完成

如果数组的“逆序对”个数较少（基本有序的），那么插入排序很好用
'''


def insertion_sort(a_need_sort):
    for i in range(1,len(a_need_sort)):  # 从下标1（第二个元素）开始，一个一个看
        tem = a_need_sort[i]  # 记录下要插入的元素值
        location = i
        while (a_need_sort[location - 1] > tem) and (location > 0):
            a_need_sort[location] = a_need_sort[location - 1]
            location -= 1
        a_need_sort[location] = tem
    return a_need_sort


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('插入排序：')
print(insertion_sort(a))
print('---------------')


'''
（4）希尔排序 —— 复杂度 O(n^(2/3))

——基本思路：利用了“插入排序”的简单性，又克服了只交换相邻元素的缺点
          定义一个增量序列 D(M) > D(M-1) > ... > D(1)=1
          对每个D(k)进行 D(k)-间隔 的排序（k=M,M-1,...,1）
          注意：每次在执行 D(k-1)间隔排序 后，这个数组的 D(k)间隔 仍然是有序的（即：每次间隔排序不影响已经排好的结果）
——过程：每隔 M 个数字，进行插入排序
       每隔 M-1 个数字，进行插入排序
       .....
       每隔 1 个数字，进行插入排序
       完成
——存在问题：如果增量元素之间 不互质 ，则小增量可能根本不起作用（8>4>2>1这种），这时的复杂度是 O(n^2)
'''


def shell_sort(a_need_sort):
    d = len(a_need_sort)//2  # 初始间隔
    while d > 0:
        # 插入排序（变形）
        for i in range(d, len(a_need_sort)):  # 从下标d开始，一个一个看
            tem = a_need_sort[i]  # 记录下要插入的元素值
            location = i
            while (a_need_sort[location - d] > tem) and (location >= d):
                a_need_sort[location] = a_need_sort[location - d]
                location -= d
            a_need_sort[location] = tem

        d = d//2  # 间隔//2
    return a_need_sort


'''改进：更多增量序列：
（1）“Hibbard”增量序列：D(k) = 2^k-1（保证相邻元素互质）
    最坏情况 —— T = O(N^(3/2))
    猜想 —— Tavg = O(N^(5/4))

（2）“Sedgewick”增量序列：{1,5,19,41,109,...}
                        —— 增量元素i = 9*4^i-9*2^i+1 或 4^i-3*2^i+1  （i从0开始）
    猜想 —— Tavg = O(N^(7/6))，Tworst = O(N^(4/3))
'''


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('希尔排序：')
print(shell_sort(a))
print('---------------')


'''
（5）堆排序 —— 复杂度 O(n*logn)

——基本思路：选择排序的升级版，不需要每次都遍历选出最小元素，利用最小堆弹出最小元素
——过程：弹出最小元素，和1交换
       弹出最小元素，和2交换
       .....
       完成
——存在问题：我只写了一个算法1，其实还有个算法2更好，但是看不太懂（具体见（2-2））
'''


def heap_sort(a_need_sort):  # 堆排序——算法1
    tem_heap = Min_Heap(len(a_need_sort))
    tem_heap.create_heap(a_need_sort)
    for i in range(len(a_need_sort)):  # 一个一个弹出
        a_need_sort[i] = tem_heap.delete_min().weight
    return a_need_sort


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('堆排序：')
print(heap_sort(a))
print('---------------')


'''
（6）归并排序 —— 复杂度 O(n*logn) —— 稳定

——基本思路（递归）：写出函数-有序子列的归并函数
                 递归调用，左边排序-右边排序-整体排序
                 若剩下一个元素，则不需要排序
——过程：略.....

——基本思路（递归）：每次归并两个相邻的子序列，直到最后全部排序完成
——过程：开始0和1归并，2和3归并，4和5归并（每两个变成整体）-存入临时数组
       整体1和整体2，整体3和整体4......-存回原数组
       .....-存入临时数组
       .....-存回原数组
       .....
       完成
       （注意：为了减少空间浪费，建立一个临时数组即可，让它和原本的数组两个之间互相倒来倒去就行）
       （注意：merge和merge_2的区别）
'''


def merge(merge_list,tem_list,left_start,right_start,right_end):  # 首先编写一个“有序子列的归并”函数
    # 认为传进来的数组，分为左右两部分，两个都分别是有序数组
    # tem_list指的是一个临时数组，用于存放最终结果
    left_end = right_start - 1  # 默认左右两个有序数组，是挨着的
    location = left_start  # 结果数组当前需要存放的位置
    merge_count = right_end - left_start + 1  # 要进行归并的元素个数
    while (left_start <= left_end) and (right_start <= right_end):  # 左右数组都不为空，循环归并
        # 开始归并，比较哪个小，就在结果数组中存哪个
        if merge_list[left_start] <= merge_list[right_start]:
            tem_list[location] = merge_list[left_start]
            location += 1  # 向后移一项
            left_start += 1
        else:
            tem_list[location] = merge_list[right_start]
            location += 1  # 向后移一项
            right_start += 1
    # 将剩下的元素依次存入结果数组（实际上，下面的两段循环，只有一个生效了）
    while left_start <= left_end:
        tem_list[location] = merge_list[left_start]
        location += 1  # 向后移一项
        left_start += 1
    while right_start <= right_end:
        tem_list[location] = merge_list[right_start]
        location += 1  # 向后移一项
        right_start += 1
    # 由于起始点一直加到最后，丢了，所以从后往前，将结果再从临时的结果数组 放回到 原数组
    for i in range(merge_count):
        merge_list[right_end] = tem_list[right_end]
        right_end -= 1


def merge_sort_1(a_need_sort,tem_list,left_start,right_end):  # 归并排序算法1——递归（分治法）
    if left_start == right_end:  # 如果只剩下一个元素了，不需要排序，直接返回
        return
    else:
        center = (left_start+right_end)//2  # 否则计算出中间值
        merge_sort_1(a_need_sort,tem_list,left_start,center)  # 左边排序一次
        merge_sort_1(a_need_sort,tem_list,center+1,right_end)  # 右边排序一次
        merge(a_need_sort,tem_list,left_start,center+1,right_end)  # 放到一起总排序


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('归并排序（递归-分治法）：')
merge_sort_1(a,[0]*len(a),0,len(a)-1)
print(a)
print('---------------')


def merge_2(merge_list,tem_list,left_start,right_start,right_end):  # 首先编写一个“有序子列的归并”函数
    # 认为传进来的数组，分为左右两部分，两个都分别是有序数组
    # tem_list指的是一个临时数组，用于存放最终结果
    left_end = right_start - 1  # 默认左右两个有序数组，是挨着的
    location = left_start  # 结果数组当前需要存放的位置
    merge_count = right_end - left_start + 1  # 要进行归并的元素个数
    while (left_start <= left_end) and (right_start <= right_end):  # 左右数组都不为空，循环归并
        # 开始归并，比较哪个小，就在结果数组中存哪个
        if merge_list[left_start] <= merge_list[right_start]:
            tem_list[location] = merge_list[left_start]
            location += 1  # 向后移一项
            left_start += 1
        else:
            tem_list[location] = merge_list[right_start]
            location += 1  # 向后移一项
            right_start += 1
    # 将剩下的元素依次存入结果数组（实际上，下面的两段循环，只有一个生效了）
    while left_start <= left_end:
        tem_list[location] = merge_list[left_start]
        location += 1  # 向后移一项
        left_start += 1
    while right_start <= right_end:
        tem_list[location] = merge_list[right_start]
        location += 1  # 向后移一项
        right_start += 1
# merge_2和merge的区别在于：这个没有最后 倒回 原数组的步骤！！！！！！（排好序的存在临时数组即可）


def merge_sort_2(a_need_sort,tem_list,n,length):  # 归并排序算法2——非递归
    # n为元素个数，length为当前归并的子列长度（默认一开始是1）
    i = 0
    while i <= n-2*length+1:  # 处理到 倒数第二对子列
        merge_2(a_need_sort,tem_list,i,i+length,i+2*length-1)  # 处理这一对
        i += 2*length  # 跳到下一对

    if i+length < n:  # 最后末尾部分还剩下超过一个子列长度，归并剩余部分
        merge_2(a_need_sort,tem_list,i,i+length,n-1)
    else:  # 否则剩下的部分少于一个子列长度，之前肯定排好的，放上去就行
        for j in range(i,n):
            tem_list[j] = a_need_sort[j]


def merge_sort_2_main(a_need_sort,n):  # 非递归 总算法程序
    length = 1
    tem = [0]*n
    while length < n:
        merge_sort_2(a_need_sort,tem,n,length)  # 排序一遍，存入临时数组
        length *= 2
        merge_sort_2(tem,a_need_sort,n,length)  # 排序一遍，存入原数组（这一步保证放在后面，这样出来之后肯定在原数组中）
        length *= 2


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('归并排序（非递归）：')
merge_sort_2_main(a,len(a))
print(a)
print('---------------')


'''
（7）快速排序 —— 复杂度 O(n*logn) —— 细节必须处理的非常到位，在大部分情况下非常快

——基本思路（递归）：分而治之
                 首先在列表中选择一个数字，作为 主元
                 将列表分为两个独立子集，左边全都 < 主元，右边全都 > 主元
                 递归处理左边、右边
                 若元素个数==1，不用处理
——细节：（1）主元怎么选？（选不对会非常慢）——最好等分数组大小——头、中、尾三个数，取中位数
       （2）怎么分独立子集？（不要浪费时间）
           在范围内从左到右扫描，判断每一个数字是否满足<pivot，否则停下，记录位置i
           在范围内从右到左扫描，判断每一个数字是否满足>pivot，否则停下，记录位置j
           两边都停下了，此时交换i、j两个位置上的数字
           重复以上，直到i > j，此时i的位置是pivot应该在的最终位置,交换right-1和i
           其中，若碰到数字=pivot，同样交换
——过程：如思路、细节
       .....
——存在问题：当数据规模较小的时候（<100），递归可能还不如插入排序快
      解决：首先正常快速排序，同时记录数据规模
           当规模到达某一个阈值（cutoff）时，停止递归，调用一个简单排序完成剩下的
'''


def select_pivot(a_need_sort,left_start,right_end):  # 选择主元函数
    center = (left_start+right_end)//2  # 计算 中 的下标
    # 经过三次比较交换，形成 left <= center <= right
    if a_need_sort[left_start] > a_need_sort[center]:
        tem = a_need_sort[left_start]
        a_need_sort[left_start] = a_need_sort[center]
        a_need_sort[center] = tem

    if a_need_sort[left_start] > a_need_sort[right_end]:
        tem = a_need_sort[left_start]
        a_need_sort[left_start] = a_need_sort[right_end]
        a_need_sort[right_end] = tem

    if a_need_sort[center] > a_need_sort[right_end]:
        tem = a_need_sort[center]
        a_need_sort[center] = a_need_sort[right_end]
        a_need_sort[right_end] = tem

    # 最后将中位数放到最右边倒数第二个位置，这样以后考虑出的范围缩小到了 left+1 ~ right-2
    tem = a_need_sort[center]
    a_need_sort[center] = a_need_sort[right_end-1]
    a_need_sort[right_end-1] = tem
    return a_need_sort[right_end-1]  # 返回主元


def insertion_sort_use(a_need_sort,left_start,n):
    for i in range(left_start+1,n):  # 从下标1（第二个元素）开始，一个一个看
        tem = a_need_sort[i]  # 记录下要插入的元素值
        location = i
        while (a_need_sort[location - 1] > tem) and (location > 0):
            a_need_sort[location] = a_need_sort[location - 1]
            location -= 1
        a_need_sort[location] = tem


def quick_sort(a_need_sort,left_start,right_end):
    # 只有一个元素，不进行排序
    if right_end-left_start >= 1:
        # 可以改成——判断阈值，若规模足够大，则进行快速排序，否则插入排序

        pivot = select_pivot(a_need_sort,left_start,right_end)  # 找到主元
        # 确定要分子集的范围
        i = left_start + 1
        j = right_end - 2

        while True:
            while a_need_sort[i] < pivot:  # 从左向右扫描，碰到不符合小于的，停止
                i += 1
            while a_need_sort[j] > pivot:  # 从右向左扫描，碰到不符合大于的，停止
                j -= 1
            if i < j:  # 若正常，代表中间还有数字没有分完，交换两边，否则退出循环
                tem = a_need_sort[i]
                a_need_sort[i] = a_need_sort[j]
                a_need_sort[j] = tem
            else:
                break
        # 退出来后，i的位置就是pivot的最终正确位置，交换
        tem = a_need_sort[i]
        a_need_sort[i] = a_need_sort[right_end-1]
        a_need_sort[right_end - 1] = tem

        quick_sort(a_need_sort,left_start,i-1)  # 递归——左边
        quick_sort(a_need_sort,i+1,right_end)   # 递归——右边

    else:
        # 可以改成——需要用到的插入排序（略改动）
        return


a = [328,3243,36,9,4,325,75,72,41,65,8,5,5,3,123,52]
print(a)
print('快速排序：')
quick_sort(a,0,len(a)-1)
print(a)
print('---------------')


'''
（8）表排序 —— 复杂度 O()

——基本思路：用于排序结构体，并非单纯的简单数字排序（比如排序几本书、字典等等）
          不移动实际数据，而移动指针
          定义一个指针数组为“表”
          每次排序，交换的只是指针数组中的元素（并非真的指针，可以用下标表示）——间接排序
          想要输出的话——A[table[0]],A[table[1]],A[table[2]].....
          （往往还配合使用物理排序，这里不写了，有点麻烦）
——过程：
'''


'''
（9）桶排序 —— 复杂度 O(M+N)

——基本思路：假设我们有 N 个学生，成绩都是介于0-100的整数，即有 M=101 个不同的成绩值
          建立一个“桶数组”，每个元素是一个链表
          一个一个扫描学生成绩，每次都将学生插入正确的桶（链表）中
          输出——桶[0]->...,桶[1]->...,桶[0]->...,......
——过程：如思路
——存在问题：若 M >> N ，怎么办，桶多人少？
'''


class student:  # 定义“学生”类——名字、成绩
    def __init__(self,name,grade):
        self.name = name
        self.grade = grade
        self.nxt = None


class bucket_list:  # 定义“桶”类
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def add_student(self,stu):  # 函数，加学生
        if self.is_empty():
            self.head = stu
        else:
            current_stu = self.head
            while current_stu.nxt is not None:
                current_stu = current_stu.nxt
            current_stu.nxt = stu


def bucket_sort(student_list,grade_list,n,m):
    count = []
    for i in range(m):  # 初始化桶数组
        count.append(bucket_list())

    for j in range(n):  # 存入n个学生信息
        tem = student(student_list[j],grade_list[j])
        count[gr[j]].add_student(tem)

    for k in range(m):  # 输出桶
        if count[k].is_empty() is False:
            current_student = count[k].head
            while current_student is not None:
                print(current_student.name,end=' ')
                print(current_student.grade)
                current_student = current_student.nxt


st = ['张三','李四','陈二','王五','刘六','关一','何七','赵八','老九']
gr = [4,7,6,5,9,8,2,3,2]
bucket_sort(st,gr,len(st),10)


'''
（10）基数排序 -- 复杂度 O(p(M+N))，p是位数（解决M >> N的情况）

——基本思路：假设我们有 N 个整数，每个整数都是介于0-999的整数，即有 M=1000 个不同的值
          建立一个“桶数组”，每个元素是一个链表
          数组的大小为10（即：十进制的基数为10，二进制的基数为2）
          由于整数都是三位数，所以我们做三次桶排序（次位优先-least significant digit）
          （1）第一次：按照个位数桶排序
          （2）第二次：在第一次的顺序基础上，按照十位数桶排序
          （3）第三次：在第二次的排序基础上，按照百位数桶排序
          输出——桶[0]->...,桶[1]->...,桶[0]->...,......
——过程：如思路
'''


class number:
    def __init__(self,num):
        self.num = num
        self.nxt = None


def radix_sort(a_need_sort):
    count_1 = []  # 个位桶排序
    count_2 = []  # 十位桶排序
    count_3 = []  # 百位桶排序
    for i in range(10):
        count_1.append(bucket_list())
        count_2.append(bucket_list())
        count_3.append(bucket_list())

    for i in range(len(a_need_sort)):  # 个位数桶排序
        tem = number(a_need_sort[i])
        gewei = a_need_sort[i] % 10  # 个位数
        count_1[gewei].add_student(tem)

    for k in range(10):  # 十位数桶排序
        if count_1[k].is_empty() is False:
            current_num = count_1[k].head
            while current_num is not None:
                tem = number(current_num.num)
                shiwei = (current_num.num // 10) % 10  # 十位数
                count_2[shiwei].add_student(tem)
                current_num = current_num.nxt

    for k in range(10):  # 百位数桶排序
        if count_2[k].is_empty() is False:
            current_num = count_2[k].head
            while current_num is not None:
                tem = number(current_num.num)
                baiwei = current_num.num // 100  # 百位数
                count_3[baiwei].add_student(tem)
                current_num = current_num.nxt

    for k in range(10):  # 输出桶
        if count_3[k].is_empty() is False:
            current_num = count_3[k].head
            while current_num is not None:
                print(current_num.num,end=' ')
                current_num = current_num.nxt


a = [64,8,216,512,27,729,0,1,343,125]
radix_sort(a)


'''
（11）多关键字排序 -- 复杂度 O(p(M+N))

——基本思路：举例扑克牌，两个关键字：四种花色（主关键字-十位），十三种数字（次关键字-个位）
          可以主位优先：四种花色对应四个桶，每个桶内调用排序算法分别排好
          可以次位优先：十三种数字对应十三个桶，运用基数排序
          输出——桶[0]->...,桶[1]->...,桶[0]->...,......
——过程：如思路
'''
