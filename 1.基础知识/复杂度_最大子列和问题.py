'''以下为最大子列和问题'''

a: list=[-2,0,4,-1,-3,20,-1,4,-8,-1,4]

'''暴力求解法（复杂度最大）  O(N^3)'''
def find_max_seq1(a):
    max_seq = 0
    ij = [0,0]
    for i in range(0, len(a)):
        for j in range(i, len(a)):
            sum_seq = 0
            for k in range(i, j+1):   # 子列每变一次，就从头计算一次
                sum_seq += a[k]
            if sum_seq >= max_seq:
                max_seq = sum_seq
                ij[0] = i
                ij[1] = j
    print(max_seq,ij)

# find_max_seq1(a)

'''算法2（减少一些复杂度）  O(N^2)'''
def find_max_seq2(a):
    max_seq = 0
    ij = [0,0]
    for i in range(0, len(a)):
        sum_seq = 0
        for j in range(i, len(a)):    # 确定子列开头后，一个元素一个元素的累加尝试，避免从头计算
            sum_seq += a[j]
            if sum_seq >= max_seq:
                max_seq = sum_seq
                ij[0] = i
                ij[1] = j
    print(max_seq,ij)

find_max_seq2(a)

'''二分法（分治法）求最大子列和  O(NlogN)'''
# 目前此方法，我不会记录下标位置
left = 0
right = len(a)-1

def binary_search(a,left,right):   # 会出现0的情况，还需要探查原因（查清楚了：最后的if判断语句问题）
    max_sum = 0
    if left == right:   # 终止递归条件
        if a[left] > 0:
            max_sum = a[left]
        else:
            max_sum = 0  # 因为如果是一个小于0的数，那么不加上去，令0即可
    else:
        center = (left + right)//2    # 求中间数，向下取整
        left_sum = binary_search(a, left, center)      # 递归求左侧最大子列和（1）
        right_sum = binary_search(a, center+1, right)  # 递归求右侧最大子列和（2）

        center_sum_left = 0
        s1 = 0
        for i in range(center,left-1,-1):    # 求跨中间数的最大子列和——左侧部分（3-1）
            s1 += a[i]
            if s1 > center_sum_left:
                center_sum_left = s1

        center_sum_right = 0
        s2 = 0
        for i in range(center+1,right+1):    # 求跨中间数的最大子列和——右侧部分（3-2）
            s2 += a[i]
            if s2 > center_sum_right:
                center_sum_right = s2

        center_sum = center_sum_left + center_sum_right  # 跨中间数的最大子列和（3）

        max_sum = center_sum    # 先令（3）为最大值

        if left_sum > max_sum:  # 判断两次，更新最大值
            max_sum = left_sum
        if right_sum > max_sum:
            max_sum = right_sum
    return max_sum

# print(binary_search(a,left,right))

'''在线处理算法  O(N)'''
# 在线：每输入一个数据就进行即时处理，在任何一个地方终止输入，算法都能正确给出当前的解
def online(a):
    max_sum = 0
    sum_seq = 0
    ij = [0, 0]
    for i in range(len(a)):
        sum_seq += a[i]         # 进来一个加一个
        if sum_seq > max_sum:   # 如果更大，更新最优解
            max_sum = sum_seq
            ij[1] = i           # 更新后，必然是队伍的末尾
        elif sum_seq < 0:       # 如果前面这一堆累加之后，变成负数了，那么对后面的累加只会起到负面作用，所以全部舍弃
            sum_seq = 0
            ij[0] = i+1         # 舍弃后，必然从下一个元素开始，作为队伍开头
    print(max_sum,ij)
# 会发现和前面算法的解，路径记录不一致，因为这是个多解题
# 在elif处，若是<=，则记录的就一样了，因为把前面等于0的累加也舍弃了

online(a)