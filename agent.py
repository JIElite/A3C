from itertools import count

import numpy
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.multiprocessing as mp

from utils import init_weights, wrap_as_variable, ensure_shared_grad, Buffer


class ActorCritic:
    def __init__(self, wid, shared_model, model, optimizer, eps_counter, result_queue, gamma=0.99, n_steps=8):
        # Configuration
        self.worker_id = wid
        self.gamma = gamma
        self.n_steps = n_steps

        # Usage data structure
        self.result_queue = result_queue
        self.eps_counter = eps_counter
        self.buffer = Buffer(size=n_steps)
        
        # network related settings
        self.global_net = shared_model
        self.local_net = model
        self.optimizer = optimizer

        # deep copy, synchronize with shared model
        self.local_net.load_state_dict(self.global_net.state_dict())

    def select_action(self, obs):
        obs = Variable(torch.from_numpy(obs).float())
        action_prob, value = self.local_net(obs)
        m = Categorical(action_prob)
        action = m.sample()
        return int(action[0].data.numpy()), m.log_prob(action), value

    def learn(self, next_state, done):
        # compute n steps approximate return G_t:t+n
        V_target = Variable(torch.FloatTensor([0.0])) if done else self.local_net(wrap_as_variable(next_state))[1]
        V_target_list = []
        for experience in self.buffer.get_reversed_experience():
            V_target = experience.reward + self.gamma* V_target
            V_target_list.insert(0, V_target)

        states = numpy.stack(self.buffer.get_n_steps_data().state)
        V_estimates = self.local_net(wrap_as_variable(states))[1]
        V_targets = torch.stack(V_target_list)

        # compute critic loss
        td_error = V_target.detach() - V_estimates
        critic_loss = td_error*td_error

        # compute actor loss
        log_action_probs = torch.stack(self.buffer.get_n_steps_data().log_action_prob)
        actor_loss = - log_action_probs*td_error.detach()
        total_loss = (actor_loss + critic_loss).mean()

        # reset gradient of local network
        self.optimizer.zero_grad()

        # loss backprobagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.local_net.parameters(), 3)

        # ensure to share the gradient with global net
        ensure_shared_grad(self.global_net, self.local_net)

        # update network parameters
        self.optimizer.step()

        # synchronize the local net with global net's parameters
        self.local_net.load_state_dict(self.global_net.state_dict())