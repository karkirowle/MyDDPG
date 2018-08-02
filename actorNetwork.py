import tensorflow as tf
import numpy as np

# Parameters from supplementary materials of the paper
layer1 = 400;
layer2 = 300;

class ActorNetwork():

    def __init__(self,state_dimension,action_dimension,name):
        self.name = name;
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        # First Hidden Layer with ReLu nonlineary
        self.state_in = tf.placeholder("float64",[None, state_dimension])
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

    def copy_weights(self,network):
        self.w1 = self.w1.assign(network.w1)
        self.b1 = self.b1.assign(network.b1)
        self.l1 = tf.nn.relu(tf.matmul(self.state_in, self.w1) + self.b1);
        self.w2 = self.w2.assign(network.w2)
        self.w3 = self.w3.assign(network.w3)
        self.b2 = self.b2.assign(network.b2)
        self.b3 = self.b3.assign(network.b3)
        self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) + self.b2);
        self.l3 = tf.tanh(tf.matmul(self.l2, self.w3) + self.b3);
            

#    @classmethod
#    def fromNetwork(self,network,name,sess):
#        
#        self.name = name;
#        # First Hidden Layer with ReLu nonlineary
#        self.state_in = tf.placeholder("float64",[None, network.state_dimension])
#        self.w1 = tf.get_variable(self.name + "w1", initializer = sess.run(
#            network.w1))
#        self.b1 = tf.get_variable(self.name + "b1", initializer = sess.run(
#            network.b1))
#        self.l1 = tf.get_variable(self.name + "l1", initializer = sess.run(
#            network.l1))
#
#        # Second Hidden Layer with ReLu nonlinearity
#        self.w2 = tf.get_variable(self.name + "w2", initializer = sess.run(
#            network.b1))
#        self.b2 = tf.get_variable(self.name + "b2", initializer = sess.run(
#            network.b2))
#        self.l2 = tf.get_variable(self.name + "w2", initializer = sess.run(
#            network.l2))
#        
#        # Last Hidden Layer with tanh nonlinearity
#        self.w3 = tf.get_variable(self.name + "w3", initializer = sess.run(
#            network.w3))
#        self.b3 = tf.get_variable(self.name + "b3", initializer = sess.run(
#            network.b3))
#        self.l3 = tf.get_variable(self.name + "l3", initializer = sess.run(
#            network.l2))
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
