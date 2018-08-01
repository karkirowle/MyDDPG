import tensorflow as tf
import numpy as np

# Parameters from supplementary materials of the paper
layer1 = 400;
layer2 = 300;

class ActorNetwork():

    def __init__(self,state_dimension,action_dimension):
        # First Hidden Layer with ReLu nonlineary
        self.state_in = tf.placeholder("float64",[None, state_dimension])
        self.w1 = faninVariables(state_dimension,layer1,"w1");
        self.b1 = faninVariables(1,layer1,"b1")
        self.l1 = tf.nn.relu(tf.matmul(self.state_in, self.w1) + self.b1);

        # Second Hidden Layer with ReLu nonlinearity
        self.w2 = faninVariables(layer1,layer2,"w2");
        self.b2 = faninVariables(1,layer2,"b2");
        self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) + self.b2);

        # Last Hidden Layer with tanh nonlinearity
        self.w3 = endVariables(layer2,action_dimension,"w3")
        self.b3 = endVariables(1,action_dimension,"b3")
        self.l3 = tf.tanh(tf.matmul(self.l2, self.w3) + self.b3);

 # Performs the square root uniform initialisation described in the Supplementary Materials
def faninVariables(dimensionx,dimensiony,name):
    return tf.get_variable(name, initializer =
                           np.random.uniform(-1/np.sqrt(dimensionx),
                                             1/np.sqrt(dimensionx),
                                             (dimensionx,dimensiony)))

 # Performs the 3*10e-3 initialisation of the end layer preceding tanh
def endVariables(dimensionx,dimensiony,name):
      return tf.get_variable(name,initializer =
                      np.random.uniform(-3*(1e-3),
                                        3*(1e-3),
                                       (dimensionx,dimensiony)))
