from net_env import NetEnvironment
from net_agent import NetAgent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def policy_train(env, agent, episode_num):
    reward_sum_line = []
    running_reward = 0
    flag = True
    P_MT_action_list = []
    P_g_action_list = []
    P_B_action_list = []
    H_D_state_list = []
    for i in range(episode_num):
        observation = env.reset(np.random.uniform(0.2, 0.9))
        reward_episode = []
        while True:
            action = agent.choose_action(np.array(observation))
            observation_, reward, done = env.step(action)
            agent.store_transition(observation, action, reward, observation_)
            reward_episode.append(reward)
            observation = observation_
            if i == episode_num - 1:
                action = env.action_space[action]
                P_MT_action_list.append(env.P_MT_action[action[0]])
                P_g_action_list.append(env.P_g_action[action[1]])
                P_B_action_list.append(env.P_B_action[action[2]])
                H_D_state_list.append(observation[6])

            if done:
                if flag:
                    running_reward = sum(reward_episode)
                    flag = False
                else:
                    running_reward = running_reward * 0.95 + sum(reward_episode) * 0.05
                reward_sum_line.append(running_reward)
                print("episode:", i + 1, "  reward:", running_reward)
                agent.learn()
                break
    reward_pd = pd.Series(reward_sum_line)
    reward_pd.to_csv('reward/reward_no_policy.csv')
    # reward_pd.to_csv('reward/reward_with_policy.csv')
    agent.save_model()
    plt.figure("奖励")
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(reward_sum_line, '-')

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


max_episode = 3000

phi = np.random.uniform(0.2, 0.9)
net_env = NetEnvironment(phi)
net_agent = NetAgent(len(net_env.action_space), 7)

policy_train(net_env, net_agent, max_episode)
