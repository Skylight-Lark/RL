import random
import gym
import numpy as np
from tqdm import tqdm
from agent import DQN
from memory import ReplayBuffer
import torch
import utils
import matplotlib.pyplot as plt


lr = 2e-3
num_iter = 10
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
seed = 123

env_name = 'Taxi-v3'
env = gym.make(env_name)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.n
action_dim = env.action_space.n

agent = DQN(state_dim, action_dim, gamma, lr,
            epsilon, target_update, device)

rewards_list = []
durations_list = []
for i in range(num_iter):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % (i+1)) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_rewards = 0
            episode_durations = 0
            state, _ = env.reset(seed=seed)
            encoded_state = utils.encode_state(state)
            done = False
            while not done:
                action = agent.take_action(encoded_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                encoded_next_state = utils.encode_state(next_state)
                
                replay_buffer.add(encoded_state, action, reward, encoded_next_state, done)
                
                state = next_state
                encoded_state = encoded_next_state
                episode_rewards += reward
                episode_durations += 1
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': state_batch,
                        'actions': action_batch,
                        'next_states': next_state_batch,
                        'rewards': reward_batch,
                        'dones': done_batch
                    }
                    agent.update(transition_dict)
            rewards_list.append(episode_rewards)
            durations_list.append(episode_durations)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(rewards_list[-10:])
                })
            pbar.update(1)
print("complete the training of DQN agent!")

agent.save(f"./model/DQN_taxi_agent.pth")

episodes_list = list(range(len(rewards_list)))
mv_rewards = utils.moving_average(rewards_list, 10)
plt.plot(episodes_list, mv_rewards)
plt.plot(episodes_list, durations_list)
plt.xlabel('Episodes')
plt.ylabel('Rewards & Durations')
plt.title('DQN on {}'.format(env_name))
plt.show()