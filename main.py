from itertools import count

import gym
import torch.multiprocessing as mp

from model import Network
from utils import SharedAdam
from agent import Worker



env = gym.make('CartPole-v0')
N_FEATURES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
LR = 2e-4


global_net = Network(N_FEATURES, N_ACTIONS)
global_net.share_memory()

optimizer = SharedAdam(global_net.parameters(), lr=LR)
optimizer.share_memory()
result_queue = mp.Queue()

workers = [Worker(env, global_net, optimizer, result_queue, N_FEATURES, N_ACTIONS, i) for i in range(mp.cpu_count())]
for worker in workers:
    worker.start()

for worker in workers:
    worker.join()
