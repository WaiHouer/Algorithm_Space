'''二分法查找元素位置'''

'''注意：查找的元素队列必须是递增的'''

a1 = [1,2,3,4,5,12,34,54,75,91]

'''非递归方法'''


def binary_search_where(a,x):
    left = 0
    right = len(a)-1
    position = -1

    while left < right:
        mid = (left + right)//2
        if a[mid] == x:         # 若找到了，那么就记录位置，跳出循环
            position = mid
            break
        elif a[mid] < x:
            left = mid + 1
        else:
            right = mid  # 最应该注意的点！！！！！！！：right 和 left 每次在取值时，应该保证+1或-1二者选一，防止死循环

    if position < 0:
        # 如果位置无效，有可能因为：
        # mid没找到，left=right正常退出循环
        # 所以还要再判断一下：
        if a[left] == x:
            position = left

    if a[position] != x:  # 若没有该元素，则返回None
        return None
    else:
        return position


print(binary_search_where(a1,54))
print(binary_search_where(a1,6))
print('-----------------------')


'''递归方法'''
left1 = 0
right1 = len(a1)-1


def binary_search_where(a, x, left, right):
    position = -1
    if left == right:  # 递归终止条件
        if a[left] == x:
            position = left
        else:
            position = None
    else:
        mid = (left + right)//2
        if a[mid] < x:
            position = binary_search_where(a, x, mid+1, right)
        elif a[mid] > x:
            position = binary_search_where(a, x, left, mid)  # 最应该注意的点！！！！！！！：right 和 left 每次在取值时，应该保证+1或-1二者选一，防止死循环
        else:
            position = mid
    return position


print(binary_search_where(a1, 54, left1, right1))
print(binary_search_where(a1, 6, left1, right1))
