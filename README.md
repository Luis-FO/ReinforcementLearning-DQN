# ReinforcementLearning-DQN
Deep Q-Learning (DQN) using PyTorch to solve reinforcement learning tasks


# Experiments

## Cart Pole (Params)

```
Melhor 'trial': 41
Melhor Métrica (Recompensa Média): 213.39
Melhores Hiperparâmetros:
{'lr': 0.00024501566707817494, 'batch_size': 64, 'gamma': 0.9867111914598459, 'tau': 0.009592709227691799, 'eps_decay': 0.9847127999191941}
```

``` python
self.net = nn.Sequential(nn.Linear(num_obs, 64), 
                            nn.ReLU(),
                            nn.Linear(64, 64), 
                            nn.ReLU(),
                            nn.Linear(64, num_actions))
```