import torch
from torch.cuda.amp import autocast, GradScaler
from model import Qnet
import numpy as np
from torch import nn
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import PIL as Image


class DQN_agent():
    ''' DQN算法 '''
    def __init__(self, state_channel, action_dim, gamma, learning_rate,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_channel, self.action_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_channel, self.action_dim).to(device) # 目标网络
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.scaler = GradScaler()
        
    def decay_epsilon(self, episode, decay_rate=0.01):
        ''' 衰减epsilon，采取指数衰减函数'''
        min_epsilon = 0.01
        max_epsilon = 1.0
        self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    def take_action(self, state, mode='train'):  
        '''epsilon-贪婪策略采取动作'''
        if mode == 'train':
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                action = self.q_net(state).argmax().item()
            return action
        elif mode == 'eval':
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
            return action

    def update(self, transition_dict):
        ''' 更新Q网络'''
        state_batch = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        action_batch = torch.tensor(transition_dict['actions'], dtype=torch.long).to(
            self.device)
        reward_batch = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).to(self.device)
        next_state_batch = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        done_batch = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).to(self.device)

        # 混合精度训练
        with autocast():
            q_values = self.q_net(state_batch).gather(1, action_batch.unsqueeze(1))  # Q值
            # 下个状态的最大Q值
            max_next_q_values = self.target_q_net(next_state_batch).max(1)[0].detach()
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)  # TD误差目标
            loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save(self, model_path):
        ''' 保存模型'''
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        
    def load(self, model_path):
        ''' 加载模型'''
        checkpoint = torch.load(model_path)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def play(self, env, save_path=None, num):
        ''' 使用训练好的模型玩一回合游戏'''
        frames = []
        state, _ = env.reset()
        done = False
        gif_file = save_path + f'play_{num}.gif'
            
        while (not done):
            img = env.render()
            frame = Image.fromarray(img)
            frames.append(frame)
            action = self.take_action(state, mode='eval')
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state

        frames[0].save(
            gif_file,
            save_all=True,
            append_images=frames[1:],
            duration=100, 
            loop=0
            )