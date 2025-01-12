import gymnasium as gym
import torch
from DQN_Agent import Agent

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


TAU = 0.05

env = gym.make('Acrobot-v1')
obs, info = env.reset()
 
num_obs = len(obs)
num_actions = env.action_space.n


agent = Agent(input_size=num_obs, output_size=num_actions, env=env, device=device)

num_episodes = 100

for i in range(num_episodes):
    if i>20:
        env = gym.make('Acrobot-v1', render_mode="human")
    done = False
    obs, info = env.reset()
    obs = torch.tensor(obs, device=device).unsqueeze(0)
    print(i)
    while not done:
        
        action = agent.select_action(obs)
        next_obs, reward, terminate, truncate, info = env.step(action.item())
        
        reward = torch.tensor([reward], device=device)
        done = terminate or truncate

        if terminate:
            next_obs = None
        else:
            next_obs = torch.tensor(next_obs, device=device).unsqueeze(0)

        agent.memory.push(obs, action, next_obs, reward)


        obs = next_obs
        agent.optimize_model()
        if done:
            break

        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()

        for key in policy_net_state_dict:

            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)




