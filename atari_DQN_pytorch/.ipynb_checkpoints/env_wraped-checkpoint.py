import cv2
from gym.core import Wrapper, ObservationWrapper
from gym.spaces import Box
import numpy as np
from collections import deque
import gym


class Preprocess(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.img_size = (84, 84)
        self.observation_space = Box(low=0, high=1.0, shape=(1, *self.img_size), dtype=np.float32)
        
    def observation(self, img):
        img = img[34:-16, :, :]
        
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img = img.astype(np.float32) / 255.0
        
        return img
    
    
class FrameStack(Wrapper):
    def __init__(self, env, n_frames=4, dim_order='pytorch'):
        super().__init__(env)
        
        self.dim_order = dim_order
        self.frames = deque([], maxlen=n_frames)
        
        n_channels, h, w, = env.observation_space.shape
        if self.dim_order == 'pytorch':
            obs_shp = [n_channels * n_frames, h, w] 
        else:
            raise ValueError('dim_order should be "pytorch"')
        self.observation_space = Box(low=0, high=1.0, shape=obs_shp, dtype=np.float32)
        
    def reset(self):
        obs, info = self.env.reset()
        for _ in range(self.frames.maxlen):
            self.frames.append(obs)
            
        return self.get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        
        return self.get_obs(), reward, terminated, truncated, info
    
    def get_obs(self):
        assert len(self.frames) == self.frames.maxlen
        if self.dim_order == 'pytorch':
            obs = np.stack(self.frames)
        else:
            raise ValueError('dim_order should be "pytorch"')
            
        return obs
    
    
def make_env(env_name, render_mode):
    env = gym.make(env_name, render_mode=render_mode)
    env = Preprocess(env)
    env = FrameStack(env)
    
    return env