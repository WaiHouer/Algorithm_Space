def build_list(m,n,num):  # 外部方法：用于建立m*n的 全num list矩阵
    ll = []
    for i in range(m):
        ll.append([])
        for j in range(n):
            ll[i].append(num)
    return ll
