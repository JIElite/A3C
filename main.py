from itertools import count
from collections import deque
import os
os.environ["OMP_NUM_THREADS"] = "1"

import gym
import torch.multiprocessing as mp

from model import Network, SeparateNetwork
from utils import SharedAdam, init_weights
from agent import ActorCritic 
from run_loop import run_loop


env = gym.make('CartPole-v0')
N_FEATURES = env.observation_space.shape[0]
LR = 5e-4
N_ACTIONS = env.action_space.n
N_STEPS = 8
NUM_WORKERS = 8
MAX_STEPS = 30000


global_net = SeparateNetwork(N_FEATURES, N_ACTIONS)
global_net.share_memory()
init_weights(global_net)

optimizer = SharedAdam(global_net.parameters(), lr=LR)
optimizer.share_memory()

# Shared Data
eps_counter = mp.Value('i', 0)

# Hogwild! style update
worker_list = []
for i in range(NUM_WORKERS):
    agent = ActorCritic(
        wid=i,
        shared_model=global_net,
        model=SeparateNetwork(N_FEATURES, N_ACTIONS),
        optimizer=optimizer,
        n_steps=N_STEPS,
    )
    worker = mp.Process(target=run_loop, args=(agent, "CartPole-v0", eps_counter, MAX_STEPS))
    worker.start()
    worker_list.append(worker)

for worker in worker_list:
    worker.join()
