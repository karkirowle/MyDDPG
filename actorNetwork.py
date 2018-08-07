import tensorflow as tf
import numpy as np

# Parameters from supplementary materials of the paper
layer1 = 400;
layer2 = 300;
learning_rate = 1e-3

class ActorNetwork():

    def __init__(self,state_dimension,action_dimension,name):

        with tf.name_scope("Actor_Network_" + name) as scope:
            self.name = name;
            self.state_dimension = state_dimension
            self.action_dimension = action_dimension
            # First Hidden Layer with ReLu nonlineary
            with tf.name_scope("Actor_Network_Layer1_"+name) as scope:
                self.state_in = tf.placeholder("float64",[None, state_dimension], name+"state_in")
                self.w1 = self.faninVariables(state_dimension,layer1,"w1");
                self.b1 = self.faninVariables(1,layer1,"b1")
                self.l1 = tf.nn.relu(tf.matmul(self.state_in, self.w1) + self.b1); 
            # Second Hidden Layer with ReLu nonlinearity
            with tf.name_scope("Actor_Network_Layer2_"+name) as scope:
                self.w2 = self.faninVariables(layer1,layer2,"w2");
                self.b2 = self.faninVariables(1,layer2,"b2");
                self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) + self.b2);

            # Last Hidden Layer with tanh nonlinearity
            with tf.name_scope("Actor_Network_Layer3_"+name) as scope:
                self.w3 = self.endVariables(layer2,action_dimension,"w3")
                self.b3 = self.endVariables(1,action_dimension,"b3")
                # Hard-coded 2 for the action interval [-2,2]
                self.l3 = tf.multiply(tf.tanh(tf.matmul(self.l2, self.w3) + self.b3), np.float64(2));

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


    def copy_network(self,tau):
        ema = tf.train.ExponentialMovingAverage(decay=1-tau)

        # This names for which variables we wish to maintain a control of EMA
        self.update = ema.apply([self.w1,self.b1,self.w2,
                            self.b2,self.w3,self.b3])

        # This obtains the reference for the shadow variables from the Ema OP
        target_net = [ema.average(x) for x in [self.w1, self.b1,
                                               self.w2,self.b2,
                                               self.w3,self.b3]]
        w1 = target_net[0]
        b1 = target_net[1]
        w2 = target_net[2]
        b2 = target_net[3]
        w3 = target_net[4]
        b3 = target_net[5]
        
        # Construct target neural network graph
        layer1 = tf.nn.relu(tf.matmul(self.state_in,w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,w2) + b2)

        # This needs to be evaluated for a feedforward run
        self.target_output = tf.multiply(tf.tanh(tf.matmul(layer2,w3) + b3),np.float64(2))

    def evaluate_target(self,sess,cur_obs):
        # Calling this evaluates the target networks weights
        return sess.run(self.target_output, feed_dict = { self.state_in: cur_obs })

    def update_target(self,sess):
        # Calling this will maintain control of new target network copy
        sess.run(self.update)


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

    # Method responsible for creating the target network using weights from another network
    def sampleAction(self,state,sess):
            return sess.run(self.l3, feed_dict = {self.state_in: state})
