import tensorflow as tf
import numpy as np

# Parameters from supplementary materials of the paper
layer1 = 400;
layer2 = 300;
learning_rate = 1e-4

class ActorNetwork():

    def __init__(self,state_dimension,action_dimension,name):

        with tf.name_scope("Actor_Network_" + name) as scope:
            self.name = name;
            self.state_dimension = state_dimension
            self.action_dimension = action_dimension
            # First Hidden Layer with ReLu nonlineary
            self.state_in = tf.placeholder("float64",[None, state_dimension], name+"state_in")
            self.w1 = self.faninVariables(state_dimension,layer1,"w1");
            self.b1 = self.faninVariables(1,layer1,"b1")
            self.l1 = tf.nn.relu(tf.matmul(self.state_in, self.w1) + self.b1);

            # Second Hidden Layer with ReLu nonlinearity
            self.w2 = self.faninVariables(layer1,layer2,"w2");
            self.b2 = self.faninVariables(1,layer2,"b2");
            self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) + self.b2);

            # Last Hidden Layer with tanh nonlinearity
            self.w3 = self.endVariables(layer2,action_dimension,"w3")
            self.b3 = self.endVariables(1,action_dimension,"b3")
            self.l3 = tf.tanh(tf.matmul(self.l2, self.w3) + self.b3);

            # Gradient optimisation - I don't completely understand what is going on here
            self.q_gradient_input = tf.placeholder("float64",
                                                   [None,action_dimension],
                                                   name+"gradient_input")
            self.parameters_gradients = tf.gradients(self.l3,
                                                     [self.w1,self.b1,
                                                      self.w2,self.w3,
                                                      self.b2,self.b3],
                                                     -self.q_gradient_input)
            self.optimiser = tf.train.AdamOptimizer(learning_rate).apply_gradients(
                zip(self.parameters_gradients,[self.w1,self.b1,
                                               self.w2,self.w3,
                                               self.b2,self.b3]))


    def copy_weights2(self,sess,network,tau):
        sess.run([self.w1.assign((1-tau)*self.w1 + tau*network.w1),
                  self.b1.assign((1-tau)*self.b1 + tau*network.b1),
                  self.w2.assign((1-tau)*self.w2 + tau*network.w2),
                  self.w3.assign((1-tau)*self.w3 + tau*network.w3),
                  self.b2.assign((1-tau)*self.b2 + tau*network.b2),
                  self.b3.assign((1-tau)*self.b3 + tau*network.b3)])
    def copy_weights(self,sess,network,tau):
        # Add tau times new, (1-tau) times old
        self.w1 = tf.Variable(sess.run((1-tau)*self.w1 +  tau*network.w1))
        self.b1 = tf.Variable(sess.run((1-tau)*self.b1 + tau*network.b1))
        self.l1 = tf.nn.relu(tf.matmul((1-tau)*self.state_in, self.w1) + self.b1);
        self.w2 = tf.Variable(sess.run((1-tau)*self.w2 + tau*network.w2))
        self.w3 = tf.Variable(sess.run((1-tau)*self.w3 + tau*network.w3))
        self.b2 = tf.Variable(sess.run((1-tau)*self.b2 + tau*network.b2))
        self.b3 = tf.Variable(sess.run((1-tau)*self.b3 + tau*network.b3))
        self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) + self.b2);
        self.l3 = tf.tanh(tf.matmul(self.l2, self.w3) + self.b3)
        sess.run(tf.variables_initializer([self.w1,self.b1,self.w2,self.w3,
                                           self.b2,self.b3]))

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

    # Method responsible for creating the target network using weights from another network
        
        
    def sampleAction(self,state,sess):
            return sess.run(self.l3, feed_dict = {self.state_in: state})
