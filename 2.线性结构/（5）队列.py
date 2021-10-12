'''队列相关基础知识'''

'''
队列：具有一定操作约束的线性表（和堆栈相反）
插入和删除：只能在一端插入，而在另一端删除
数据插入：入队列（AddQ），rear记录位置
数据删除：出队列（DeleteQ），front记录位置
先来先服务
先进先出：FIFO
'''
'''采用循环队列方法：存到头了，但是前面由于删除操作还有很多空位，那么新元素回过头继续从头开始存'''
'''具体参考课件！'''

'''简单的队列（规定了大小，所以需要循环队列///不规定大小的话贼简单）'''


class Queue:
    def __init__(self,size):  # 初始化队列
        self.size = size
        self.queue = [None]*size
        self.front = -1  # 记录队伍开头（末尾进，开头出）
        self.rear = -1   # 记录队伍末尾（末尾进，开头出）

    def add_queue(self,data):  # 方法：入队列
        if (self.rear + 1) % self.size == self.front:  # 靠尾部+1取size的余数，判断队列满不满
            print('队列已满，无法入队列')
        else:
            self.rear = (self.rear + 1) % self.size  # 队尾后移一位
            self.queue[self.rear] = data

    def delete_queue(self):  # 方法：出队列
        if self.front == self.rear:
            print('队列为空，无法出队列')
        else:
            self.front = (self.front + 1) % self.size  # 同理上面
            need_to_delete = self.queue[self.front]  # 临时变量，记录出队列的数据
            self.queue[self.front] = None  # 删掉出完的位置
            print(need_to_delete)
            return need_to_delete

    def length(self):  # 方法：计算队列长度（即，队列中元素个数）
        length = self.size - self.queue.count(None)
        return length


q = Queue(5+1)
for i in range(5):
    q.add_queue(i)

q.delete_queue()
q.delete_queue()
q.delete_queue()

q.add_queue(39)
q.add_queue(7)
q.add_queue(333)
q.add_queue(85)

for i in range(q.length()):
    q.delete_queue()