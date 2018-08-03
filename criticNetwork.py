import tensorflow as tf
import numpy as np

# Parameters from supplementary materials of the paper
layer1 = 400;
layer2 = 300;
learning_rate = 1e-3
beta = 1e-2
class CriticNetwork():
    def __init__(self,state_dimension,action_dimension,name):

        # Make appending global
        self.name = name;
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        
        # First Hidden Layer with ReLu nonlineary
        self.state_in = tf.placeholder("float64",[None, state_dimension], name+"state_in")
        self.w1 = self.faninVariables(state_dimension,layer1,"w1");
        self.b1 = self.faninVariables(1,layer1,"b1")
        self.l1 = tf.nn.relu(tf.matmul(self.state_in, self.w1) + self.b1);

        # Second Hidden Layer with ReLu nonlinearity and action input introduced
        self.w2 = self.faninVariables(layer1,layer2,"w2");
        self.b2 = self.faninVariables(1,layer2,"b2");
        self.action_in = tf.placeholder("float64",[None,action_dimension], name+"action_in");
        self.actionw1 = self.faninVariables(action_dimension,layer2,"actionw1")
        self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) +
                             tf.matmul(self.action_in,self.actionw1) +
                             self.b2);

        # Last Hidden Layer which is unbounded to ensure sufficient expressivity
        self.w3 = self.endVariables(layer2,1,"w3")
        self.b3 = self.endVariables(1,1,"b3")
        self.l3 = tf.identity(tf.matmul(self.l2, self.w3) + self.b3);

        # Optimisation objective
        self.Y = tf.placeholder("float64",[None,1], name+"Y")
        cost = tf.reduce_mean(self.Y - self.l3 + beta*(tf.nn.l2_loss(self.w1) +
                              tf.nn.l2_loss(self.b1) +
                              tf.nn.l2_loss(self.w2) +
                              tf.nn.l2_loss(self.b2) +
                              tf.nn.l2_loss(self.w3) +
                              tf.nn.l2_loss(self.b3) +
                                                       tf.nn.l2_loss(self.actionw1)));
        self.optimiser = tf.train.AdamOptimizer(1e-3).minimize(cost);
        self.critic_gradients = tf.gradients(self.l3,self.action_in)[0];
        
    def copy_weights(self,sess,network,tau):
        self.w1 = tf.Variable(sess.run(tau*self.w1 + (1-tau)*network.w1))
        self.b1 = tf.Variable(sess.run(tau*self.b1 + (1-tau)*network.b1))
        self.l1 = tf.nn.relu(tf.matmul(self.state_in, self.w1) + self.b1);
        self.w2 = tf.Variable(sess.run(tau*self.w2 + (1-tau)*network.w2))
        self.w3 = tf.Variable(sess.run(tau*self.w3 + (1-tau)*network.w3))
        self.b2 = tf.Variable(sess.run(tau*self.b2 + (1-tau)*network.b2))
        self.b3 = tf.Variable(sess.run(tau*self.b3 + (1-tau)*network.b3))
        self.actionw1 = tf.Variable(sess.run(self.actionw1 + (1-tau)*network.actionw1))
        self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) +
                             tf.matmul(self.action_in,self.actionw1) +
                             self.b2);
        self.l3 = tf.identity(tf.matmul(self.l2, self.w3) + self.b3)

        sess.run(tf.variables_initializer([self.w1,self.b1,self.w2,
                                           self.w3,self.b2,self.b3,self.actionw1]))

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
                               np.random.uniform(-3*(1e-3),
                                                 3*(1e-3),
                                                 (dimensionx,dimensiony)))

    def sampleValue(self,sess,state,action):
        return sess.run(self.l3, feed_dict = {self.state_in: state, self.action_in: action})

    def optimiseCritic(self,sess,y,states,actions,learning_rate):
        Y = tf.Variable(y)
        cost = tf.reduce_mean(Y - self.l3)/len(y);
        optimisation = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        sess.run(optimisation,feed_dict = {self.state_in: states, self.action_in: actions})
        print("Opti done")
