from itertools import count
from collections import deque
import os

import gym
import torch.multiprocessing as mp

from model import Network, SeparateNetwork
from utils import SharedAdam, init_weights, xavier_init
from agent import ActorCritic 
from run_loop import run_loop


os.environ["OMP_NUM_THREADS"] = "1"


def evaluate(eps_counter, result_queue):
    # TODO # of eps 不見得是抓到對的，可能會跳號！一次跳了 200 或是重複計數
    evaluation_queue = deque(maxlen=100)
    while True:
        evaluation_queue.append(result_queue.get())
        if eps_counter.value > 0 and eps_counter.value % 100 == 0:
            print("recently eps avg rewards:", sum(evaluation_queue) / evaluation_queue.maxlen, 
            "# of eps:", eps_counter.value)
        

env = gym.make('CartPole-v0')
N_FEATURES = env.observation_space.shape[0]
LR = 5e-4
N_ACTIONS = env.action_space.n
N_STEPS = 8
NUM_WORKERS = 8
MAX_STEPS = 30000


global_net = SeparateNetwork(N_FEATURES, N_ACTIONS)
global_net.share_memory()
xavier_init(global_net)

optimizer = SharedAdam(global_net.parameters(), lr=LR)
optimizer.share_memory()

# Shared Data
eps_counter = mp.Value('i', 0)
result_queue = mp.Queue()

# Hogwild! style update
worker_list = []
for i in range(NUM_WORKERS):
    agent = ActorCritic(
        wid=i,
        shared_model=global_net,
        model=SeparateNetwork(N_FEATURES, N_ACTIONS),
        optimizer=optimizer,
        eps_counter=eps_counter,
        result_queue=result_queue,
        n_steps=N_STEPS,
    )
    worker = mp.Process(target=run_loop, args=(agent, "CartPole-v0", MAX_STEPS))
    worker.start()
    worker_list.append(worker)
for worker in workers:
    worker.join()