import tensorflow as tf
import numpy as np

import gym

from actorNetwork import ActorNetwork
from criticNetwork import CriticNetwork
# DDPG Implementation

env = gym.make('CartPole-v0')
observation = env.reset()


# Quick pseudocodish view 

# Target actor
#targetActor = ActorNetwork(3,5,)
targetCritic = CriticNetwork(3,10,"b")

# Non-target
#actor = ActorNetwork(3,5,"c")
critic = CriticNetwork(3,10,"d")
