import tensorflow as tf
import numpy as np

import gym

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
total_episodes = 10000
action_dimension = 1 # This is determined by the pendulum environment
state_dimension = 3 # Pendulum env
mini_batch = 64 # minibatch transition
gamma = 0.99 # from supplementary materials
buffer_size = 1000000 # from supplementary
full_flag = False
test_frequency = 10

# Create a new environment
env = gym.make('Pendulum-v0')
observation = env.reset()
total_steps = env.spec.timestep_limit

# Replay Buffer
replay_buffer = np.empty(buffer_size,dtype=ReplayBuffer);
k = 0;

# Quick pseudocodish view 

# Non-target
actor = ActorNetwork(state_dimension,action_dimension,"realActor")
critic = CriticNetwork(state_dimension,action_dimension,"realCritic")

# Initialise target neural networks
target_actor = ActorNetwork(state_dimension,action_dimension,"targetActor")
target_critic = CriticNetwork(state_dimension,action_dimension,"targetCritic")
# Initialise session that we will pass around as global variable
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Copy the weights from the non-target initialisations
target_actor.copy_weights(sess,actor,tau=0);
target_critic.copy_weights(sess,critic,tau=0);

for episode in range(0,total_episodes):
    # Initialise random process
    noise = Noise(action_dimension)
    for steps in range(0,total_steps):
        
        # Sample an action from policy neural network
        action = actor.sampleAction(
            observation.reshape(1,state_dimension),sess) + noise.sampleNoise(action_dimension,1);
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
    if full_flag:
        # Sample the indices of steps so that we have i.i.d mini-batch
        indices = np.random.permutation(buffer_size)
        sampled_buffer = [replay_buffer[i] for i in indices[0:mini_batch]];
        y = np.zeros((mini_batch,1))
        for i in range(0,mini_batch):
            # Unfolding tuple for easier interpretation
            cur_old_obs = sampled_buffer[i].state.reshape(1,state_dimension)
            cur_reward = sampled_buffer[i].reward
            cur_action = sampled_buffer[i].action.reshape(1,action_dimension)
            cur_obs = sampled_buffer[i].new_state.reshape(1,state_dimension)
            target_action = target_actor.sampleAction(cur_obs,sess)
            # Creating target for loss function
            y[i] = cur_reward + gamma*target_critic.sampleValue(sess,cur_obs, target_action)
            states = np.array([sampled_buffer[i].state.reshape(
                state_dimension) for i in range(0,mini_batch)])
            actions = np.array([sampled_buffer[i].action.reshape(
                action_dimension) for i in range(0,mini_batch)])
            # Train critic
            sess.run(critic.optimiser, feed_dict = {critic.Y: y.reshape(mini_batch,1),
                                            critic.action_in: actions.reshape(mini_batch,1),
                                            critic.state_in: states.reshape(mini_batch,3)})
            critic_gradients = critic.obtain_gradients(sess, actions, states);
            sess.run(actor.optimiser,
             feed_dict = { actor.q_gradient_input: critic_gradients,
                           actor.state_in : states})
            target_actor.copy_weights(sess,actor,tau=0.001)
            target_critic.copy_weights(sess,critic,tau=0.001)
            print("Training, Step:", step, "Episode:",episode)
    # Test after every test episode for learning curve TODO
    if (episode % test_frequency == 0):
        sum_reward = 0
        for steps in range(0,total_steps):
            # Sample an action from policy neural network now without noise
            action = actor.sampleAction(observation.reshape(1,state_dimension),sess);
            observation, reward, _, _ = env.step(action)
            env.render()
            sum_reward = sum_reward + reward
        # Code snippet for plotting the change of reward with TensorBoard
        summary_writer = tf.summary.FileWriter("logdir")
        summary = tf.Summary()
        summary.value.add(tag="SumOfReward", simple_value = sum_reward)
        summary_writer.add_summary(summary, episode)
        summary_writer.flush() 
                
# Closing the Tensorflow session, tautological comments ftw<3
sess.close()
