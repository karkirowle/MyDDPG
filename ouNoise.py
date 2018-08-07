import numpy as np


class Noise():

    def __init__(self,action_dimension, mu = 0, theta = 0.15, sigma = 0.2, noise_scale = 1):
        self.x = np.random.normal((action_dimension))
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dimension = action_dimension
        self.noise_scale = 1;
        
    def sampleNoise(self,action_dimension,samples):
        self.x = self.theta * (self.mu-self.x) + self.sigma*np.random.randn(action_dimension);
        #self.x += self.noise_scale*dx;
        return self.x;
