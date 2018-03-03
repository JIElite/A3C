from itertools import count

import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.multiprocessing as mp


from model import Network
from utils import init_weights, wrap_as_variable, ensure_shared_grad, Buffer


class Worker(mp.Process):
    def __init__(self, env, gnet, optimizer, result_queue, n_features, n_actions, wid, gamma=0.99, n_steps=8, max_steps=10000):
        super(Worker, self).__init__()
        # Configuration
        self.worker_id = wid
        self.gamma = gamma
        self.n_steps = n_steps
        self.max_steps = max_steps
        self.env = env
        
        # Usage data structure
        self.result_queue = result_queue
        self.buffer = Buffer(size=n_steps)
        
        # network related settings
        self.global_net = gnet
        self.local_net = Network(n_features, n_actions)
        self.optimizer = optimizer
        init_weights(self.local_net)

    def select_action(self, obs):
        obs = Variable(torch.from_numpy(obs).float())
        action_prob, value = self.local_net(obs)
        m = Categorical(action_prob)
        action = m.sample()
        return int(action[0].data.numpy()), m.log_prob(action), value

    def learn(self, next_state, done):
        # compute n steps approximate return G_t:t+n
        V_target = Variable(torch.FloatTensor([0.0])) if done else self.local_net(wrap_as_variable(next_state))[1]
        for experience in self.buffer.get_reversed_experience():
            V_target = experience.reward + self.gamma* V_target
    
        V_pi = self.local_net(wrap_as_variable(self.buffer.memory[0].state))[1]

        # compute loss
        td_error = V_target - V_pi
        critic_loss = td_error.pow(2)
        critic_loss.data.clamp_(0, 20)

        action = self.buffer.memory[0].action
        log_action_prob = self.local_net(wrap_as_variable(self.buffer.memory[0].state))[0][action]
        actor_loss = - log_action_prob * td_error.detach()
        # print(actor_loss, critic_loss)
        total_loss = actor_loss + 0.5*critic_loss

        # reset gradient of local network
        self.optimizer.zero_grad()
        # loss backprobagation
        total_loss.backward()
        # ensure to share the gradient with global net
        ensure_shared_grad(self.global_net, self.local_net)
        # update network parameters
        self.optimizer.step()
        # synchronize the local net with global net's parameters
        self.local_net.load_state_dict(self.global_net.state_dict())

    def run(self):
        obs = self.env.reset()
        eps_reward = 0

        for step in range(1, self.max_steps+1):
            action, log_action_prob, value = self.select_action(obs)
            obs_, reward, done, _ = self.env.step(action)
            eps_reward += reward

            self.buffer.append([obs, action, log_action_prob, reward])
            if self.buffer.is_full() or done:
                self.learn(obs_, done)

            # transition
            obs_ = obs
            if done:
                print('work no: {} step: {}, reward: {}'.format(self.worker_id, step, eps_reward))
                self.result_queue.put(eps_reward)
                eps_reward = 0
                self.buffer.reset()
                obs = self.env.reset()

                

        