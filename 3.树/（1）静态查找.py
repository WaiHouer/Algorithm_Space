'''以下介绍静态查找的相关知识'''


'''方法1：顺序查找（有“哨兵”）'''


class Node_type:   # 定义一个节点，存放了想要查找的数组
    def __init__(self,a):
        a.insert(0,'k')  # 在该数组的最开头，放置一个“哨兵”，“哨兵”等于退出循环的标志
        self.a = a


class List:
    def __init__(self):
        self.head = None

    def search_object(self,data_list):
        data_object = Node_type(data_list)
        self.head = data_object  # 让指针指向想要查找的这个数组


def sequential_search(ob, x):  # 顺序查找函数
    position = 0  # 初始化下标位置是0
    for i in range(len(ob.head.a)-1,-1,-1):  # 倒序查找
        if ob.head.a[i] != 'k':  # 直到碰见“哨兵”，意味着退出循环
            if ob.head.a[i] == x:
                position = i  # 若在碰见“哨兵”之前找到了该数字，就记录下标位置
        else:
            break
    if position:  # 下标有效就输出，无效就代表没有
        print(f'第{position}个数字是{x}')
    else:
        print('在列表中没有要查找的数字')


ll = List()
ll.search_object([2,3,4,57,3,43,76,87,34,432])
sequential_search(ll, 4)
sequential_search(ll, 33333)


'''方法2：二分查找（请参考小算法中的py文件）'''
