from core.base_class import BaseAlgorithm


class OnPolicyAlgorithm(BaseAlgorithm):
    def __init__(self, env, learning_rate, device):
        super().__init__(env, learning_rate, device)


    def learn(self):
        pass