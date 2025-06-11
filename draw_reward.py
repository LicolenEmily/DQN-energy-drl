import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def get_reward_data(path):
    df_reward = pd.read_csv(path)
    col_data_reward = df_reward.iloc[:, 1]
    col_data_reward = list(col_data_reward)[0:]
    return col_data_reward


def draw_figure_1():
    reward_no_policy = get_reward_data('reward/reward_no_policy.csv')
    reward_with_policy = get_reward_data('reward/reward_with_policy.csv')
    plt.figure(1)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(reward_no_policy, color='red', label="无策略奖励")
    plt.plot(reward_with_policy, color='blue', label="有策略奖励")
    pl.legend()
    plt.show()


draw_figure_1()
# self.observation = [self.P_PV[0], self.P_WT[0], self.L_e[0], self.L_h[0], self.L_c[0], self.price[0],np.random.uniform(0.2, 0.9)]
