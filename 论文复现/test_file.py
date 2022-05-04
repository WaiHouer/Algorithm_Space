for b in self.b:
    print(f'b（每周新增病床数）：{b}')
    S_0_pre = [0 for i in range(self.region_num)]  # 滚动预测起点（为拟合的最后一天）
    E_0_pre = [0 for i in range(self.region_num)]
    A_0_pre = [0 for i in range(self.region_num)]
    Q_0_pre = [0 for i in range(self.region_num)]
    U_0_pre = [0 for i in range(self.region_num)]
    R_0_pre = [0 for i in range(self.region_num)]
    D_0_pre = [0 for i in range(self.region_num)]
    for i in range(self.region_num):
        S_0_pre[i] = S_fit[i][-1]
        E_0_pre[i] = E_fit[i][-1]
        A_0_pre[i] = A_fit[i][-1]
        Q_0_pre[i] = Q_fit[i][-1]
        U_0_pre[i] = U_fit[i][-1]
        R_0_pre[i] = R_fit[i][-1]
        D_0_pre[i] = D_fit[i][-1]

    start = self.end

    B_before = 0
    a_before = [0 for i in range(self.region_num)]

    for t in range(self.predict_num):  # （2）对各仿真节点进行迭代规划
        if t % 7 == 0:
            b_t = b
        else:
            b_t = 0
        print(f'迭代仿真第 {t + 1} / {self.predict_num} 天')

        # （2-1）混合整数规划模型，确定最佳的a值
        model = LP(self.region_num, self.total_population, B_before, a_before, b_t, b * self.r, self.ksi,
                   self.simulate_para_list[t], self.dist, S_0_pre, A_0_pre, U_0_pre)

        self.a[t] = model.a  # 写入a值

        # （2-2）代入a值，利用new传染病模型，仿真预测出下一期
        simulation_a = SEAQURD(self.region_num, self.file_name, start, start + 1, self.total_population,
                               S_0_pre, E_0_pre, A_0_pre, Q_0_pre, U_0_pre, R_0_pre, D_0_pre,
                               self.simulate_para_list[t], self.ksi, self.a[t])
        for i in range(self.region_num):  # 存入结果
            # self.simulate_a_result[i][t] = simulation_a.I[i][-1] - simulation_a.Q[i][-1]  # Q是已经入院的
            self.simulate_a_result[i][t] = simulation_a.E[i][-1]  # Q是已经入院的

        # 更新起点（即向前推进一天）
        start += 1
        for i in range(self.region_num):
            S_0_pre[i] = simulation_a.S[i][-1]
            E_0_pre[i] = simulation_a.E[i][-1]
            A_0_pre[i] = simulation_a.A[i][-1]
            Q_0_pre[i] = simulation_a.Q[i][-1]
            U_0_pre[i] = simulation_a.U[i][-1]
            R_0_pre[i] = simulation_a.R[i][-1]
            D_0_pre[i] = simulation_a.D[i][-1]

        # 更新B_before和a_before
        B_before = model.B
        a_before = model.a

    for t in range(1, self.predict_num):
        for i in range(self.region_num):  # 对净增长人数进行逆运算
            self.a_addition[f'b={b}'] += self.simulate_a_result[i][t] - self.simulate_a_result[i][t - 1] \
                                         + self.simulate_para_list[t][3][i] * self.simulate_a_result[i][t - 1]
