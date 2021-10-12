'''完美立方数小算法'''

'''
a^3 = b^3 + c^3 + d^3 成为完美立方数
其中b<=c<=d
本算法要求输出a<N的所有完美立方数组合
'''

def perfect_cube(N):
    for a in range(2,N+1):  # 2到N
        for b in range(2,a):  # 2到a-1，a没必要
            for c in range(b,a):
                for d in range(c,a):
                    if a**3 == b**3+c**3+d**3:
                        print('a= ',a,' ','b,c,d= ',b,c,d)


N = 24
perfect_cube(N)