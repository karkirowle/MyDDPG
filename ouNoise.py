import numpy as np


class Noise():

    def __init__(self,action_dimension):
        self.x = np.random.normal((action_dimension))
    def sampleNoise(self,action_dimension,samples, mu = 0, theta = 0.15, sigma = 0.2):
        dx = theta * (mu-self.x) + sigma*np.random.normal(action_dimension);
        self.x += dx;
        return self.x;
