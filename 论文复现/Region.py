'''
由于要多区域建模，所以建立“区域”类
'''


class Region:
    def __init__(self,name,location,N,S,E,A,Q,U,R,D):
        self.name = name
        self.location = location  # 区域坐标：用于计算区域间距离（也可能用不上）
        self.N = N
        self.S = S
        self.E = E
        self.A = A
        self.Q = Q
        self.U = U
        self.R = R
        self.D = D

    def show_info(self):
        print(f'区域“{self.name}”，坐标：{self.location}')
        print(f'N：{self.N}，S：{self.S}，E：{self.E}，A：{self.A}，Q：{self.Q}，U：{self.U}，R：{self.R}，D：{self.D}')


if __name__ == '__main__':
    TEM = Region('武汉',[2,2],10000,10000,0,0,0,0,0,0)
    TEM.show_info()
