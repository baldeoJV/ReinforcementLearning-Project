from agent import Agent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import TransformObservation, ResizeObservation

import ale_py
import cv2  

def to_grayscale(obs):
    return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


episodes = 10000
max_episode_steps = 10000
hidden_layer = 128
learning_rate = 0.0001
step_repeat = 4
gamma = 0.99
batch_size = 64
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.995


env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

env = ResizeObservation(env, (64, 64))

env = GrayscaleObservation(env,  keep_dim=True)

# env = TransformObservation(env,  f=to_grayscale)

agent = Agent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate, step_repeat=step_repeat,
              gamma=gamma)

agent.test()