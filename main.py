import tensorflow as tf
import numpy as np

import gym

from actorNetwork import ActorNetwork
from criticNetwork import CriticNetwork
from ouNoise import Noise
# DDPG Implementation

# Parameters
total_episodes = 2
total_steps = 10
action_dimension = 3 # This is determined by the pendulum environment
state_dimension = 10 # TODO: Revise
mini_batch = 5 # minibatch transition

# Create a new environment
env = gym.make('Pendulum-v0')
observation = env.reset()

# Replay Buffer
replay_buffer = [];

# Quick pseudocodish view 



# Non-target
actor = ActorNetwork(action_dimension,1,"real")
critic = CriticNetwork(state_dimension,action_dimension,"realCritic")


# Copy weights to target neural networks
# target_actor = ActorNetwork.fromNetwork(actor,"target",sess)
target_actor = ActorNetwork(action_dimension,1,"targetActor")
target_actor.copy_weights(actor);
target_critic = CriticNetwork(state_dimension,action_dimension,"targetCritic")
target_critic.copy_weights(critic);
# Initialise session that we will pass around as global variable
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for episode in range(0,total_episodes):

    # Initialise random process
    noise = Noise(action_dimension)
    for steps in range(0,total_steps):
        
        # Sample an action from policy neural network
        action = actor.sampleAction(
            observation.reshape(1,action_dimension),sess) + noise.sampleNoise(action_dimension,1);
        print(action)
        old_observation = observation;
        observation, reward, _, _ = env.step(action)
        env.render()
        replay_buffer.append( (old_observation,reward,action,observation) )

    # Sample the indices of steps so that we have i.i.d mini-batch
    indices = np.random.permutation(steps)
    sampled_buffer = [replay_buffer[i] for i in indices[0:mini_batch]];
    print(len(sampled_buffer))

    # Feedforward value sampling from the target critic


sess.close()
