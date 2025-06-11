from net_env import NetEnvironment
from net_agent import NetAgent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def policy_test(env, agent):
    flag = True
    P_MT_action_list = []
    P_g_action_list = []
    P_B_action_list = []
    H_D_state_list = []

    observation = env.reset(0.6)
    reward_episode = []
    while True:
        action = agent.choose_best_action(np.array(observation))
        observation_, reward, done = env.step(action)
        reward_episode.append(reward)
        observation = observation_
        action = env.action_space[action]
        P_MT_action_list.append(env.P_MT_action[action[0]])
        P_g_action_list.append(env.P_g_action[action[1]])
        P_B_action_list.append(env.P_B_action[action[2]])
        H_D_state_list.append(observation[6])
        if done:
            break
    plt.figure("联合发电单元功率")
    plt.xlabel('time')
    plt.ylabel('功率')
    plt.plot(P_MT_action_list, '-')

    plt.figure("电网流入微能源网的电功率")
    plt.ylabel('功率')
    plt.xlabel('time')
    plt.plot(P_g_action_list, '-')

    plt.figure("蓄电池充放电功率")
    plt.xlabel('time')
    plt.ylabel('功率')
    plt.plot(P_B_action_list, '-')

    plt.figure("蓄电池核电状态")
    plt.xlabel('time')
    plt.ylabel('核电状态')
    plt.plot(H_D_state_list, '-')

    plt.show()


net_env = NetEnvironment(0.5)
net_agent = NetAgent(len(net_env.action_space), 7)
policy_test(net_env, net_agent)
