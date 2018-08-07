import tensorflow as tf
import numpy as np

import gym

import time

from actorNetwork import ActorNetwork
from criticNetwork import CriticNetwork
from ouNoise import Noise

class ReplayBuffer():
    "Stores the replay experience"
    def __init__(self,state,action,reward,new_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state

# DDPG Implementation

# Parameters
total_episodes = 100000000
action_dimension = 1 # This is determined by the pendulum environment
state_dimension = 3 # Pendulum env
mini_batch = 64 # minibatch transition
gamma = 0.99 # from supplementary materials
buffer_size = 100000 # --from supplementary-- custom
full_flag = False
test_frequency = 1
train_frequency = 1
tau = 0.001


# Create a new environment
env = gym.make('Pendulum-v0')
total_steps = env.spec.timestep_limit

# Replay Buffer
replay_buffer = np.empty(buffer_size,dtype=ReplayBuffer);
k = 0;

# Non-target
actor = ActorNetwork(state_dimension,action_dimension,"a")
critic = CriticNetwork(state_dimension,action_dimension,"b")

# Initialise target neural networks inside the actor-critic instances
actor.copy_network(tau)
critic.copy_network(tau)

# Initialise session that we will pass around as global variable
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Let me draaww
summary_writer = tf.summary.FileWriter("logdir")
summary_writer.add_graph(sess.graph)

# Merging summaries from copied networks
merge = tf.summary.merge_all()

print(tf.trainable_variables())
for episode in range(0,total_episodes):
    # Reset environment
    observation = env.reset() 
    # Noise scale
    noise_scale = (0.1 * 0.99**episode) * (env.action_space.high - env.action_space.low)
    
    # Initialise random process
    noise = Noise(action_dimension, mu=0, theta = 0.15, sigma=0.2, noise_scale=noise_scale)

    for steps in range(0,total_steps):

        # Sample an action from policy neural network
        noiseSample = noise.sampleNoise(action_dimension,1)
        action = actor.sampleAction(observation.reshape(1,state_dimension),sess) + noiseSample;
        old_observation = observation;
        observation, reward, done, _ = env.step(action)
        env.render()

        # Fill buffer incrementally
        replay_buffer[k] = ReplayBuffer(old_observation,reward,action,observation)
        if k < (buffer_size-1):
            k = k + 1
        else:
            k = 0
            full_flag = True
            
        # Start training only if the buffer is full
        if (full_flag and (steps % train_frequency == 0)):

            # Sample the indices of steps so that we have i.i.d mini-batch
            sampled_buffer = np.random.choice(replay_buffer,mini_batch)

            y = np.array([sampled_buffer[i].reward +
                 gamma*critic.evaluate_target(sess,
                                              sampled_buffer[i].state.reshape(
                                                  1,
                                                  state_dimension),
                                              actor.evaluate_target(
                                                  sess,
                                                  sampled_buffer[i].state.reshape(
                                                      1,
                                                      state_dimension)))
                 for i in range (0,mini_batch)])
                                                                    
            states = np.array([sampled_buffer[i].state.reshape(
                    state_dimension) for i in range(0,mini_batch)])
            actions = np.array([sampled_buffer[i].action.reshape(
                    action_dimension) for i in range(0,mini_batch)])

            # Train critic
            sess.run(critic.optimiser,
                                 feed_dict = {critic.Y: y.reshape(mini_batch,1),
                                              critic.action_in: actions.reshape(mini_batch,
                                                                                action_dimension),
                                              critic.state_in: states.reshape(mini_batch,
                                                                              state_dimension)})
            critic_gradients = critic.obtain_gradients(sess, actions, states);
                
            # Train actor
            sess.run(actor.optimiser,
                     feed_dict = { actor.q_gradient_input: critic_gradients,
                                   actor.state_in : states})
            newSum = sess.run(merge, feed_dict = {critic.Y: y.reshape(mini_batch,1),
                                                  critic.action_in: actions.reshape(mini_batch,
                                                                                    action_dimension),
                                                  critic.state_in: states.reshape(mini_batch,
                                                                                  state_dimension),
                                                  actor.state_in : states,
                                                  actor.q_gradient_input: critic_gradients})
            
            # Perform exponential moving average update with prescribed tau
            actor.update_target(sess)
            critic.update_target(sess)
            
            if done:
                break

    # Add last summary of the episode, to have one sample for each
    # Test after every test episode for learning curve 
    if ((episode % test_frequency == 0) and full_flag):
        print("This function is called in", episode, " episode and", steps, "step")
        summary_writer.add_summary(newSum,episode)
        sum_reward = 0
        observation = env.reset()
        max_value = -99999999999999 # no idea what would be a sensible lower bound
        for steps in range(0,total_steps):
            # Sample an action from policy neural network now without noise
            action = actor.sampleAction(observation.reshape(1,state_dimension),sess);
            value = critic.sampleValue(sess,observation.reshape(1,state_dimension),
                                       action.reshape(1,action_dimension));
            observation, reward, done, _ = env.step(action)
            env.render()
            sum_reward = sum_reward + reward
            max_value = np.maximum(max_value,value)
            if done:
                break

        # Constructing a new summary message with relevant scalars and adding it
        summary = tf.Summary()
        summary.value.add(tag="SumOfReward", simple_value = sum_reward)
        summary.value.add(tag="MaxValue", simple_value= max_value)
        summary_writer.add_summary(summary, episode)
        summary_writer.flush() 
                
# Closing the Tensorflow session, tautological comments ftw<3
sess.close()
