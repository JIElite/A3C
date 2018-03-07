import gym


def run_loop(agent, env_id, max_steps=30000):
    env = gym.make(env_id) 
    obs = env.reset()
    eps_reward = 0

    for step in range(1, max_steps+1):
        action, log_action_prob, value = agent.select_action(obs)
        obs_, reward, done, _ = env.step(action)
        eps_reward += reward

        agent.buffer.append([obs, action, log_action_prob, reward])
        if agent.buffer.is_full() or done:
            agent.learn(obs_, done)
            agent.buffer.reset()

        # transition
        obs = obs_
        if done:
            print('work no: {} eps: {}, reward: {}'.format(agent.worker_id, agent.eps_counter.value, eps_reward))
            agent.result_queue.put(eps_reward)
            agent.eps_counter.value += 1
            eps_reward = 0
            agent.buffer.reset()
            obs = env.reset()
    
