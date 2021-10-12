'''一些小例子'''

# 顺序输出1——N的整数（循环or递归）
# N = int(input('输入N值'))
def xunhuan(N):        # 循环方法
    for i in range(N):
        print(i + 1)

def digui(N):        # 递归方法
    if N:
        digui(N-1)
        print(N)

# xunhuan(N)
# digui(N)


# 计算多项式 f(x)=a0 + a1*x + a2*x^2 + a3*x^3 + ...+ an-1*x^n-1 + an*x^n
a: list=[1,2]
x = 2
def zhijiesuan(a,x):
    p = 0
    for i in range(len(a)):
        p += a[i]*(x**i)
    return p

def zhuanhuan(a,x):   # 多项式变成f(x)=a0 + x(a1 + x(a2+ x(....(an-1 + x(an)))))
    p = a[len(a)-1]
    for i in range(len(a)):
        p = a[-i-1] + p*x
        print(p)                # 有问题！！！先不修了
    return p

# print(zhijiesuan(a,x))
# print(zhuanhuan(a,x))

