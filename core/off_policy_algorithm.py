from core.base_class import BaseAlgorithm
import numpy as np

class OffPolicyAlgorithm(BaseAlgorithm):
    def __init__(self, env, device, warmup_steps=1000):
        super().__init__(env, device)

        self.memory = None
        self.warmup_steps = warmup_steps

    def on_train_start(self):
        pass

    def on_step(self):
        pass

    def _store_transition(self, obs, action, reward, next_obs, done):
        self.memory.push(obs, action, reward, next_obs, done)

    def setEnv(self, env):
        self.env = env

    def get_info(self):
        pass

    def train(self, total_steps):
        episode_rewards = []

        self.on_train_start()

        # Training loop
        steps = 0
        episode = 0
        while steps < total_steps:
            obs, _ = self.env.reset()
            done = False
            total_reward = 0 
            # Loop until the end of the episode or until total_steps is reached
            while not done and steps < total_steps:
                if steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    # Select action with epsilon-greedy policy.
                    action = self.select_action(obs, training = True)
                

                # Execute action in the environment
                next_obs, reward, terminate, truncate, _ = self.env.step(action)
                
                done = terminate or truncate

                # Store the transition in the replay memory
                self._store_transition(obs, action, reward, next_obs, done)

                # Update the agent after each step if outside the warmup period
                if steps >= self.warmup_steps:
                    self._update()
                
                obs = next_obs
                total_reward += reward
                steps+=1
                self.on_step()

            # TODO: Change print to logging and include more info like loss, etc.
            if steps >= self.warmup_steps:
                info = self.get_info()

                print(f"Episode {episode}: Reward = {total_reward:.2f} {info}")
            else:
                status = "Warm Up" if steps < self.warmup_steps else "Training"
                print(f"Episode {episode}: Reward {total_reward:.2f} | Status: {status} | Total Steps: {steps}")

            episode_rewards.append(total_reward)
            episode += 1

        self.env.close()
        
        if len(episode_rewards) >= 100:
            final_metric = sum(episode_rewards[-100:]) / 100
        else:
            # Case the training is too short or interrupted
            final_metric = sum(episode_rewards) / len(episode_rewards) if len(episode_rewards) > 0 else -1000

        print(f"Training completed. Final metric (average of 100): {final_metric}")
        return final_metric
        
    def test(self, num_episodes):
        episode_rewards = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(obs, training=False)
                next_obs, reward, terminate, truncate, _ = self.env.step(action)

                done = terminate or truncate
                obs = next_obs
                total_reward += reward

            episode_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}/{num_episodes}: Reward = {total_reward:.2f}")

        self.env.close()

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        print(f"\n--- Test Completed ---")
        print(f"Mean: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward