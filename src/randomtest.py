"""Test class"""

# import random

from env import CoinGame

env = CoinGame(3, 3)
env.reset(seed=10)
env.render()
env.close()

# for i in range(5):
# for agent in env.agent_iter():
# observation, reward, done, _ = env.last()
# act = env.legal_moves()
# action = None if done else random.choice(act)
# env.render()
# env.step(action)
# print(agent)
# print("reward : ")
# print(reward)
# env.render()
# i = i + 1
