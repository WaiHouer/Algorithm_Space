'''堆栈相关基础知识'''

'''
中缀表达式：运算符号位于两个运算数中间，如：a + b * c - d / e
后缀表达式：运算符号位于两个运算数中间，如：a b c * + d e / -

计算机识别出后缀表达式，然后求解

后缀表达式求值策略：从左向右扫描，逐个处理（1）遇到运算数，先记下来（2）遇到运算符号，找到最后记住的两个数进行运算

于是，需要这种“倒序”的数据结构，即：堆栈

插入数据：入栈（push）
删除数据：出栈（pop）
后入先出：Last In First Out（LIFO）
'''

'''！！！！！！   List   可以直接当堆栈用'''

'''简单的单向堆栈（不规定大小，后续可以自己添加规定大小的功能）'''


class Stack:
    def __init__(self):  # 初始化堆栈
        self.stack = []

    def is_empty(self):  # 方法：是否为空
        return self.stack == []

    def size(self):  # 方法：查看堆栈大小
        return len(self.stack)

    def push(self,data):  # 方法：入栈
        self.stack.append(data)

    def pop(self):  # 方法：出栈
        if not self.stack:
            print('堆栈为空，没有元素可以出栈')
        else:
            return self.stack.pop()

    def peek(self):  # 方法：查看顶部元素（并不删除）
        return self.stack[self.size()-1]


a = Stack()
a.push(1)
a.push(65)
a.push(35)
a.push(941)
a.push(4)
print(a.size())
print('顶部元素是：',a.peek())
for i in range(a.size()):
    print(a.pop())


print('----------------------')
'''具有规定大小的，一个数组存放两个堆栈'''
'''具体方法：为了让空间没有浪费，两个堆栈从数组的两端开始向中间增长'''


class Double_Stack:
    def __init__(self,size):  # 初始化堆栈（指定大小）
        self.size = size
        self.stack = [None] * size
        self.top1 = -1    # 记录双侧堆栈顶端位置
        self.top2 = size

    def is_empty(self):  # 方法：判断是否为空
        if self.top1 == -1:
            print('堆栈1为空',end='')
        else:
            print('堆栈1不为空',end='')
        if self.top2 == self.size:
            print('堆栈2为空')
        else:
            print('堆栈2不为空')

    def push(self,tag,data):  # 方法：tag为标签，指定第几个堆栈，入栈
        if self.top2-self.top1 == 1:
            print('堆栈已满，不能再入栈')
        elif tag == 1:
            self.top1 += 1   # 注意：要先这样进一位top值，再进行入栈
            self.stack[self.top1] = data
        else:
            self.top2 -= 1
            self.stack[self.top2] = data

    def pop(self,tag):  # 方法：tag为标签，指定第几个堆栈，出栈
        if tag == 1:
            if self.top1 == -1:
                print('堆栈1为空，没有元素可以出栈')
            else:
                self.top1 -= 1   # 注意：要先这样后退一位top值，为了记录顶端变化
                print(self.stack[self.top1+1])
                return self.stack[self.top1+1]  # 再在下面进一位上来，进行入栈
        else:
            if self.top2 == self.size:
                print('堆栈1为空，没有元素可以出栈')
            else:
                self.top2 += 1
                print(self.stack[self.top2-1])
                return self.stack[self.top2-1]


b = Double_Stack(10)
b.is_empty()
for i in range(1,4):
    b.push(1,i)
b.is_empty()
for j in range(1,8):
    b.push(2,j)
b.is_empty()
print('堆栈1pop：')
for i in range(b.top1+1):
    b.pop(1)
print('堆栈2pop：')
for j in range(b.size-b.top2):
    b.pop(2)
