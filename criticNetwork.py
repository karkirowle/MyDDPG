import tensorflow as tf
import numpy as np

# Parameters from supplementary materials of the paper
layer1 = 400;
layer2 = 300;
learning_rate = 1e-3
beta = 1e-2
class CriticNetwork():
    def __init__(self,state_dimension,action_dimension,name):

        with tf.name_scope("Critic_Network_" + name) as scope:
            # Make appending global
            self.name = name;
            self.state_dimension = state_dimension
            self.action_dimension = action_dimension

            
            # First Hidden Layer with ReLu nonlineary
            self.state_in = tf.placeholder("float64",[None, state_dimension], name+"state_in")
            with tf.name_scope("Critic_Network_Layer_1" + name) as scope:
                self.w1 = self.faninVariables(state_dimension,layer1,"w1");
                tf.summary.histogram(self.w1.op.name, self.w1)
                self.b1 = self.faninVariables(1,layer1,"b1")
                tf.summary.histogram(self.b1.op.name, self.b1)
                self.l1 = tf.nn.relu(tf.matmul(self.state_in, self.w1) + self.b1);
                tf.summary.histogram(self.l1.op.name, self.l1)
            # Second Hidden Layer with ReLu nonlinearity and action input introduced
            with tf.name_scope("Critic_Network_Layer_2" + name) as scope:
                self.w2 = self.faninVariables(layer1,layer2,"w2");
                tf.summary.histogram(self.w2.op.name, self.w2)
                self.b2 = self.faninVariables(1,layer2,"b2");
                tf.summary.histogram(self.b2.op.name, self.b2)
                self.action_in = tf.placeholder("float64",[None,action_dimension], name+"action_in");
                self.actionw1 = self.faninVariables(action_dimension,layer2,"actionw1")
                tf.summary.histogram(self.actionw1.op.name, self.actionw1)
                self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) +
                                     tf.matmul(self.action_in,self.actionw1) +
                                     self.b2);

            # Last Hidden Layer which is unbounded to ensure sufficient expressivity
            with tf.name_scope("Critic_Network_Layer_3" + name) as scope:
                self.w3 = self.endVariables(layer2,1,"w3")
                tf.summary.histogram(self.w3.op.name, self.w3)
                self.b3 = self.endVariables(1,1,"b3")
                tf.summary.histogram(self.b3.op.name, self.b3)
                self.l3 = tf.identity(tf.matmul(self.l2, self.w3) + self.b3);
                tf.summary.histogram(self.l3.op.name, self.l3)
            # Optimisation objective
            self.Y = tf.placeholder("float64",[None,1], name+"Y")
            cost = tf.reduce_mean(tf.square(self.Y - self.l3) +
                                  beta*(tf.nn.l2_loss(self.w1) +
                                        tf.nn.l2_loss(self.b1) +
                                        tf.nn.l2_loss(self.w2) +
                                        tf.nn.l2_loss(self.b2) +
                                        tf.nn.l2_loss(self.w3) +
                                        tf.nn.l2_loss(self.b3) +
                                        tf.nn.l2_loss(self.actionw1)));
            self.optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cost);
            self.critic_gradients = tf.gradients(self.l3,self.action_in)[0];

    def copy_network(self,tau):
        ema = tf.train.ExponentialMovingAverage(decay=1-tau)

        # This names for which variables we wish to maintain a control of EMA
        self.update = ema.apply([self.w1,self.b1,self.w2,self.b2,
                            self.actionw1,self.w3,self.b3])

        # This does an instant update, copying the original weights of the NN
        target_net = [ema.average(x) for x in [self.w1, self.b1,
                                               self.w2,self.b2,
                                               self.actionw1,
                                               self.w3,self.b3]]

        # Explicitly written out for ease of understanding
        w1 = target_net[0]
        b1 = target_net[1]
        w2 = target_net[2]
        b2 = target_net[3]
        actionw1 = target_net[4]
        w3 = target_net[5]
        b3 = target_net[6]
        
        # Construct target neural network graph
        layer1 = tf.nn.relu(tf.matmul(self.state_in,w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,w2) +
                            tf.matmul(self.action_in,actionw1) +
                                      b2)

        # This needs to be evaluated for a feedforward run
        self.target_output = tf.identity(tf.matmul(layer2,w3) + b3)

    def evaluate_target(self,sess,cur_obs,target_action):
        # Calling this will maintain control of new target network copy
        return sess.run(self.target_output, feed_dict = { self.state_in: cur_obs,
                                                   self.action_in: target_action })
    def update_target(self,sess):
        # Calling this will maintain control of new target network copy
        sess.run(self.update)

    # Obtains the gradient for the actor's optimiser
    def obtain_gradients(self,sess,actions,states):
        return sess.run(self.critic_gradients,
                        feed_dict = { self.state_in : states,
                                      self.action_in : actions})
    
# Performs the square root uniform initialisation described in the Supplementary Materials
    def faninVariables(self,dimensionx,dimensiony,name):
        return tf.get_variable(self.name + name, initializer =
                               np.random.uniform(-1/np.sqrt(dimensionx),
                                                 1/np.sqrt(dimensionx),
                                                 (dimensionx,dimensiony)))

    # Performs the 3*10e-3 initialisation of the end layer preceding tanh
    def endVariables(self,dimensionx,dimensiony,name):
        return tf.get_variable(self.name + name,initializer =
                               np.random.uniform(-3e-3,
                                                 3e-3,
                                                 (dimensionx,dimensiony)))

    def sampleValue(self,sess,state,action):
        return sess.run(self.l3, feed_dict = {self.state_in: state, self.action_in: action})

