import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
import collections
import random
import copy
from typing import Tuple # For Python < 3.9
import matplotlib.pyplot as plt
import pandas as pd


class DQN_Agent():
    """由于环境已将离散动作重新定义为{0,1}，我们可以简单地用一个数字表示动作。"""

    def __init__(self,
                 Q_func: torch.nn.Module,  # Q_func 参数应该是一个 PyTorch 神经网络模型的实例
                 action_dim: int,  # 动作空间的维度
                 optimizer: torch.optim.Optimizer,  # 优化器，用于更新神经网络模型的参数
                 replay_buffer: collections.deque,  # 经验回放缓冲区，用于存储智能体在环境中的经验
                 replay_start_size: int,  # 经验回放缓冲区开始进行训练的最小经验数量
                 batch_size: int,  # 每次从经验回放缓冲区中抽样的批量大小
                 replay_frequent: int,  # 多少步进行一次经验回放
                 target_sync_frequent: int,  # 两个Q网络参数同步的频率
                 epsilon: float = 0.1,  # ε-greedy算法的初始探索率
                 mini_epsilon: float = 0.01,  # ε-greedy算法的最小探索率
                 explore_decay_rate: float = 0.0001,  # ε-greedy算法的探索率衰减率
                 gamma: float = 0.9,  # 折扣因子，用于计算未来奖励的衰减值
                 device: torch.device = torch.device("cpu")  # 运行模型的设备，默认是 CPU
                 ) -> None:

        self.device = device  # 保存运行模型的设备，可以是 CPU 或 GPU
        self.action_dim = action_dim  # 保存动作空间的维度

        self.exp_counter = 0  # 经验计数器，用于跟踪经验的数量

        self.replay_buffer = replay_buffer  # 经验回放缓冲区，用于存储智能体在环境中的经验
        self.replay_start_size = replay_start_size  # 经验回放缓冲区开始进行训练的最小经验数量
        self.batch_size = batch_size  # 每次从经验回放缓冲区中抽样的批量大小
        self.replay_frequent = replay_frequent  # 多少步进行一次经验回放

        self.target_sync_frequent = target_sync_frequent  # 两个Q网络参数同步的频率

        """使用两个Q函数（main_Q和target_Q）来稳定训练过程。
           由于它们共享相同的网络结构，我们可以使用copy.deepcopy将main_Q复制到target_Q进行初始化。"""
        self.main_Q_func = Q_func  # 主Q函数，用于选择动作
        self.target_Q_func = copy.deepcopy(Q_func)  # 目标Q函数，用于计算目标值

        self.optimizer = optimizer  # 优化器，用于更新神经网络模型的参数

        self.epsilon = epsilon  # ε-greedy算法的初始探索率
        self.mini_epsilon = mini_epsilon  # ε-greedy算法的最小探索率
        self.gamma = gamma  # 折扣因子，用于计算未来奖励的衰减值
        self.explore_decay_rate = explore_decay_rate  # ε-greedy算法的探索率衰减率


    pass

    def get_target_action(self, obs: np.ndarray) -> int:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)  # 将观察值转换为 PyTorch 张量，并设置设备
        Q_list = self.target_Q_func(obs)  # 通过目标Q函数获取状态(obs)对应的动作值列表
        action = torch.argmax(Q_list).item()  # 选择具有最大动作值的动作
        return action  # 返回选择的动作


    def get_behavior_action(self, obs: np.ndarray) -> int:
        """这里使用简单的epsilon衰减来平衡探索和利用。
           epsilon从epsilon_init衰减到mini_epsilon。"""
        self.epsilon = max(self.mini_epsilon, self.epsilon - self.explore_decay_rate)  # 使用简单的epsilon衰减策略

        if np.random.uniform(0, 1) < self.epsilon:  # 根据当前epsilon值随机探索
            action = np.random.choice(self.action_dim)  # 随机选择动作
        else:
            action = self.get_target_action(obs)  # 根据目标Q值选择动作

        return action  # 返回选择的动作


    """这里，我们定义了一个函数来同步main_Q和target_Q的参数。"""

    def sync_target_Q_func(self) -> None:
        # 同步目标Q函数和主Q函数的参数
        for target_params, main_params in zip(self.target_Q_func.parameters(), self.main_Q_func.parameters()):
            target_params.data.copy_(main_params.data)

    def batch_Q_approximation(self,
                              batch_obs: torch.tensor,
                              batch_action: torch.tensor,
                              batch_reward: torch.tensor,
                              batch_next_obs: torch.tensor,
                              batch_done: torch.tensor) -> None:
        # 计算批量Q值的近似
        batch_current_Q = torch.gather(self.main_Q_func(batch_obs), 1, batch_action).squeeze(1)  # 获取当前状态下选择动作的Q值
        batch_TD_target = batch_reward + (1 - batch_done) * self.gamma * self.target_Q_func(batch_next_obs).max(1)[0]  # 计算TD目标值
        loss = torch.mean(F.mse_loss(batch_current_Q, batch_TD_target))  # 计算均方误差损失

        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 使用优化器更新神经网络参数

    def Q_approximation(self,
                        obs: np.ndarray,
                        action: int,
                        reward: float,
                        next_obs: np.ndarray,
                        done: bool) -> None:

        self.exp_counter += 1  # 增加经验计数器

        # 将当前经验添加到经验回放缓冲区
        self.replay_buffer.append((obs, action, reward, next_obs, done))

        # 如果经验回放缓冲区中的经验数量超过了阈值，并且经验计数达到了经验回放的频率
        if len(self.replay_buffer) > self.replay_start_size and self.exp_counter % self.replay_frequent == 0:
            # 从经验回放缓冲区中抽样一批数据进行Q值的训练
            self.batch_Q_approximation(*self.replay_buffer.sample(self.batch_size))

        # 每 target_sync_frequent 步同步两个Q网络的参数
        if self.exp_counter % self.target_sync_frequent == 0:
            self.sync_target_Q_func()  # 同步目标Q函数和主Q函数的参数



class Q_Network(torch.nn.Module):
    """在这里定义自己的网络结构。"""

    def __init__(self, obs_dim: int, action_dim) -> None:
        super(Q_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)  # 输入层到第一个隐藏层，64个神经元
        self.fc2 = torch.nn.Linear(64, action_dim)  # 第一个隐藏层到输出层，输出动作空间维度个神经元

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = F.relu(x)  # 使用ReLU激活函数
        x = self.fc2(x)

        return x


class ReplayBuffer():
    def __init__(self, capacity: int, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)  # 用于存储经验数据的循环缓冲区

    def append(self, exp_data: tuple) -> None:
        self.buffer.append(exp_data)  # 将经验数据添加到缓冲区中

    def sample(self, batch_size: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        mini_batch = random.sample(self.buffer, batch_size)  # 从缓冲区中随机抽样形成一个mini-batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)

        obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)

        action_batch = torch.tensor(action_batch, dtype=torch.int64, device=self.device)
        action_batch = action_batch.unsqueeze(1)  # 将动作扩展为列向量

        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        next_obs_batch = torch.tensor(np.array(next_obs_batch), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self) -> int:
        return len(self.buffer)  # 返回缓冲区中的经验数量


class TrainManager():

    def __init__(self,
                 env: gym.Env,
                 episode_num: int = 1000,
                 lr: float = 1e-3,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 mini_epsilon: float = 0.01,
                 explore_decay_rate: float = 0.0001,
                 buffer_capacity: int = 2000,
                 replay_start_size: int = 200,
                 replay_frequent: int = 4,
                 target_sync_frequent: int = 200,
                 batch_size: int = 32,
                 seed: int = 0,
                 my_device: str = "cpu"
                 ) -> None:
        """
        初始化训练管理器

        参数:
            env (gym.Env): OpenAI Gym环境
            episode_num (int): 训练的总轮数
            lr (float): 学习率
            gamma (float): 折扣因子
            epsilon (float): 探索-利用策略的初始探索概率
            mini_epsilon (float): 探索-利用策略的最小探索概率
            explore_decay_rate (float): 探索概率衰减率
            buffer_capacity (int): 经验回放缓冲区的容量
            replay_start_size (int): 开始训练前经验回放缓冲区中的最小经验数量
            replay_frequent (int): 每隔多少步进行一次经验回放
            target_sync_frequent (int): 每隔多少步同步一次目标网络
            batch_size (int): 每次从经验回放缓冲区中采样的批量大小
            seed (int): 随机数生成器的种子
            my_device (str): 使用的设备，"cpu" 或 "cuda"

        返回:
            None
        """
        # 设置随机数生成器的种子，确保实验的可复现性
        self.episode_num = episode_num
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

        # 设置设备（CPU 或 GPU）
        self.device = torch.device(my_device)

        # 初始化环境和重置环境状态
        self.env = env
        _, _ = self.env.reset(seed=self.seed)

        # 初始化训练相关参数和模型
        obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        action_dim = env.action_space.n
        self.buffer = ReplayBuffer(capacity=buffer_capacity, device=self.device)
        Q_func = Q_Network(obs_dim, action_dim).to(self.device)
        optimizer = torch.optim.Adam(Q_func.parameters(), lr=lr)
        self.agent = DQN_Agent(Q_func=Q_func,
                               action_dim=action_dim,
                               optimizer=optimizer,
                               replay_buffer=self.buffer,
                               replay_start_size=replay_start_size,
                               batch_size=batch_size,
                               replay_frequent=replay_frequent,
                               target_sync_frequent=target_sync_frequent,
                               epsilon=epsilon,
                               mini_epsilon=mini_epsilon,
                               explore_decay_rate=explore_decay_rate,
                               gamma=gamma,
                               device=self.device)

        # 初始化用于存储每轮训练总奖励的数组
        self.episode_total_rewards = np.zeros(episode_num)
        self.index_episode = 0

    def train_episode(self) -> float:
        """
        训练一个回合的方法

        返回:
            total_reward (float): 本回合的总奖励
        """
        total_reward = 0
        obs, _ = self.env.reset()
        while True:
            # 从代理获取动作
            action = self.agent.get_behavior_action(obs)

            # 执行动作，获取下一个状态、奖励和终止信息
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 使用 Q 网络进行 Q 值的近似
            self.agent.Q_approximation(obs, action, reward, next_obs, done)

            obs = next_obs
            if done:
                # 存储本回合的总奖励
                self.episode_total_rewards[self.index_episode] = total_reward
                self.index_episode += 1
                break

        return total_reward

    def train(self) -> None:
        """
        训练方法，循环执行多个回合的训练

        返回:
            None
        """
        for e in range(self.episode_num):
            # 训练一个回合
            episode_reward = self.train_episode()

            # 每100回合打印一次总奖励
            if e % 100 == 0:
                print('Episode %s: Total Reward = %.2f' % (e, episode_reward))

    def plotting(self, smoothing_window: int = 100) -> None:
        """绘制随时间变化的每个episode的奖励值。"""
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.episode_total_rewards, label="Episode Reward")
        # 使用滑动平均对曲线进行平滑
        rewards_smoothed = pd.Series(self.episode_total_rewards).rolling(smoothing_window,
                                                                         min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed, label="Episode Reward (Smoothed)")
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title("随时间变化的Episode奖励值")
        plt.legend()
        plt.show()


# 创建CartPole环境
env = gym.make('CartPole-v0')

# 创建训练管理器对象
Manager = TrainManager(env=env,
                      episode_num=2000,
                      lr=1e-3,
                      gamma=0.9,
                      epsilon=0.3,
                      target_sync_frequent=200,
                      mini_epsilon=0.1,
                      explore_decay_rate=0.0001,
                      seed=0,
                      my_device="cpu"
                      )

# 进行深度强化学习的训练
Manager.train()

# 绘制训练过程中的奖励曲线
Manager.plotting()
