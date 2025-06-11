import numpy as np


# 光伏的功率输出
def get_P_PV():
    mean_value = [0, 0, 0, 0, 0, 5, 12, 20, 25, 28, 30, 32, 30, 28, 25, 18, 12, 6, 3, 0, 0, 0, 0, 0]
    P_PV = []
    for item in mean_value:
        P_PV.append(float(np.random.normal(item, 0.1 * item, 1)[0]))
    return P_PV


# 风机的功率输出
def get_P_WT():
    mean_value = [12, 20, 21, 22, 23, 24, 25, 23, 21, 18, 25, 24, 23, 21, 20, 8, 2, 0, 0, 0, 0, 15, 23, 18]
    P_WT = []
    for item in mean_value:
        P_WT.append(float(np.random.normal(item, 0.15 * item, 1)[0]))
    return P_WT


# 电负荷
def get_L_e():
    mean_value = [0, 0, 0, 0, 0, 0, 0, 110, 110, 115, 110, 110, 110, 110, 110, 110, 120, 60, 35, 20, 15, 0, 0, 0]
    L_e = []
    for item in mean_value:
        L_e.append(float(np.random.normal(item, 0.05 * item, 1)[0]))
    return L_e


# 热负荷
def get_L_h():
    mean_value = [0, 0, 0, 0, 0, 0, 0, 85, 130, 145, 145, 145, 145, 150, 165, 175, 170, 165, 110, 90, 80, 0, 0, 0]
    L_h = []
    for item in mean_value:
        L_h.append(float(np.random.normal(item, 0.05 * item, 1)[0]))
    return L_h


# 冷负荷
def get_L_c():
    mean_value = [0, 0, 0, 0, 0, 0, 0, 30, 60, 55, 70, 75, 65, 60, 65, 50, 50, 30, 20, 15, 20, 0, 0, 0]
    L_c = []
    for item in mean_value:
        L_c.append(float(np.random.normal(item, 0.05 * item, 1)[0]))
    return L_c


# 电价
def get_price():
    price = [0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.86, 0.86, 0.86, 0.86, 0.86, 1.1, 1.1, 1.1, 1.1, 0.86,
             0.86, 1.1, 1.1, 1.1, 0.86, 0.86]
    return price


class NetEnvironment:
    def __init__(self, phi):
        self.eta_MT = 0.3  # 联供发电单元的发电效率
        self.eta_HR = 0.73  # 余热回收锅炉的换热效率
        self.eta_HX = 0.9  # 换热装置的换热效率
        self.eta_SB = 0.9  # 燃气锅炉的效率
        self.eta_EC = 4  # 电制冷机的性能系数

        self.eta_BC = 0.2  # 电储能的最大充电率
        self.eta_BD = 0.4  # 电储能的最大放电率
        self.eta_B_max = 0.9  # 电储能的最大储能系数
        self.eta_B_min = 0.2  # 电储能的最小储能系数

        self.P_WT_max = 30  # 风机最大功率
        self.P_PV_max = 35  # 光伏最大功率

        self.P_PV = get_P_PV()  # 光伏的功率输出
        self.P_WT = get_P_WT()  # 风机的功率输出
        self.L_e = get_L_e()  # 电负荷
        self.L_h = get_L_h()  # 热负荷
        self.L_c = get_L_c()  # 冷负荷
        self.price = get_price()  # 电价

        self.P_MT_action = [0, 15, 30, 45, 60]  # 联合发电单元功率(0-60)
        self.P_g_action = [0, 16, 32, 48, 64, 80]  # 电网流入微能源网的电功率
        self.P_B_action = [-40, -20, 0, 20, 40, 60, 80]  # 蓄电池充放电功率
        self.action_space = []
        self.W_B = 200  # 蓄电池的最大容量
        self.c_f = 3.45  # 天然气的单位价格
        self.q_ng = 9.7  # 天然气的低热值(kW h)/m3
        self.Q_SB_max = 100  # 联供发电单元输出的最大热功率
        self.Q_HR_max = 120  # 余热回收锅炉输出的最大热功率
        self.P_M_t_1 = 30
        self.c_p = 0.9  # 单位差额电量的惩罚
        self.observation = [self.P_PV[0], self.P_WT[0], self.L_e[0], self.L_h[0], self.L_c[0], self.price[0],
                            phi]  # 光伏发电，风力发电、电负荷、热负荷、电负荷、电价、电储能的荷电状态
        self.c_b = 10

        self.d_t = 1
        self.T = 24
        self.t = 0

        for i in range(len(self.P_MT_action)):
            for j in range(len(self.P_g_action)):
                for k in range(len(self.P_B_action)):
                    temp = [i, j, k]
                    self.action_space.append(temp)

    def reset(self, phi):
        self.t = 0
        self.P_PV = get_P_PV()  # 光伏的功率输出
        self.P_WT = get_P_WT()  # 风机的功率输出
        self.L_e = get_L_e()  # 电负荷
        self.L_h = get_L_h()  # 热负荷
        self.L_c = get_L_c()  # 冷负荷
        self.price = get_price()  # 电价
        self.observation = [self.P_PV[0], self.P_WT[0], self.L_e[0], self.L_h[0], self.L_c[0], self.price[0], phi]
        return self.observation

    def get_observation_reward(self, action):

        # 获取t时段分时电价
        gamma_t = self.price[self.t]
        # 获取t时段内的平均购电功率
        P_g_t = self.P_g_action[self.action_space[action][1]]
        # 当前时刻的购电成本
        C_e = gamma_t * P_g_t * self.d_t

        # 1、联供发电单元模型
        # 获取当前时刻的联供发电单元的发电功率
        P_MT_t = self.P_g_action[self.action_space[action][0]]
        # 根据当前时刻的联供发电单元的发电功率，计算联供发电单元的单位时间天然气消耗量
        V_MT_t = P_MT_t / (self.q_ng * self.eta_MT)
        # 联供发电单元的单位时间天然气消耗量,计算联供发电单元的输出热功率
        Q_MT_t = V_MT_t * self.q_ng * (1 - self.eta_MT)

        # 2、余热回收锅炉模型
        # 计算余热回收锅炉的输出热功率
        Q_HR_t = self.eta_HR * Q_MT_t
        if Q_HR_t > self.Q_HR_max:
            Q_HR_t = self.Q_HR_max

        # 3、燃气锅炉模型
        # 获取当前时刻的热负荷
        L_h_t = self.L_h[self.t]
        # 热力总线能量平衡方程,计算出燃气锅炉的输出热功率
        Q_SB_t = L_h_t / self.eta_HX - Q_HR_t
        if Q_SB_t > self.Q_SB_max:
            Q_SB_t = self.Q_SB_max
        # 根据燃气锅炉的输出热功率,计算燃气锅炉的单位时间天然气消耗量
        V_SB_t = Q_SB_t / (self.q_ng * self.eta_SB)

        # 计算当前时刻的购入天然气成本
        C_f = self.c_f * (V_MT_t + V_SB_t)

        # 4、换热装置模型
        # 计算换热装置的输出功率。换热装置将联供发电单元和和燃气锅炉产生的热能进行转化
        Q_HX_t = (Q_HR_t + Q_SB_t) * self.eta_HX

        # 5、电制冷机模型
        # 获取当前时段的冷负荷
        L_c_t = self.L_c[self.t]

        # 电计算电制冷机的制冷功率
        P_EC_t = L_c_t / self.eta_EC

        # 5、电储能模型
        P_B_t = self.P_B_action[self.action_space[action][2]]
        P_BD_t, P_BC_t = 0, 0
        if P_B_t > 0:
            # 放电功率
            P_BD_t = P_B_t
        else:
            # 充电功率
            P_BC_t = abs(P_B_t)

        observation_phi = self.observation[6]
        next_observation_phi = observation_phi + (P_BC_t * self.d_t - P_BD_t * self.d_t) / self.W_B

        # 获取当前时刻的风机的发电功率
        P_WT_t = self.P_WT[self.t]

        # 获取当前时刻光伏的发电功率
        P_PV_t = self.P_PV[self.t]

        # 获取当前时刻的电负荷消耗
        L_e_t = self.L_e[self.t]

        # 电力总线不平衡电量和
        d_p_i = abs(P_MT_t + P_BD_t + P_g_t + P_PV_t + P_WT_t - L_e_t - P_BC_t - P_EC_t)
        # 计算电力总线不平衡电量的惩罚
        D_P_i = self.c_p * d_p_i

        # 计算电池过放或过充电量的惩罚
        W_b_t = self.W_B * observation_phi
        W_b_t_max = self.W_B * self.eta_B_max
        W_b_t_min = self.W_B * self.eta_B_min
        d_b_i = 0
        if P_B_t > 0:
            # 放电功率
            temp_w_b = W_b_t - P_BD_t * self.d_t
            if temp_w_b < W_b_t_min:
                d_b_i = abs(W_b_t_min - temp_w_b)
                next_observation_phi = self.eta_B_min
        else:
            # 充电功率
            temp_w_b = W_b_t + P_BC_t * self.d_t
            if temp_w_b > W_b_t_max:
                d_b_i = abs(temp_w_b - W_b_t_max)
                next_observation_phi = self.eta_B_max
        D_B_i = d_b_i * self.c_p

        temp_e_i = (P_MT_t - self.P_M_t_1) / 60
        if temp_e_i > 0.5:
            D_E_i = abs(temp_e_i - 0.5) * self.c_b
        else:
            D_E_i = abs(temp_e_i + 0.5) * self.c_b
        self.t += 1
        next_observation = [self.P_PV[self.t], self.P_WT[self.t], self.L_e[self.t], self.L_h[self.t], self.L_c[self.t],
                            self.price[self.t], next_observation_phi]

        reward = -(C_e + C_f + D_P_i + D_B_i + D_E_i)
        return next_observation, reward

    def step(self, action):
        is_done = False
        next_observation, reward = self.get_observation_reward(action)
        if self.t >= self.T - 1:
            is_done = True
        return next_observation, reward, is_done
