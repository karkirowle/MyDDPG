
class ActorNetwork(object):

    def create_actor_network(self):
        inputs = tflearn.input_data(
            shape=[None, self.s_dim])
        net = tflearn.fully_connected(
            inputs,400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net,300)
        net = tflearn.layers.normalization.batch_normalization(net)
        # Final layer weights are init to uniform
        w_init = tflearn.initializations.uniform(
            minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh',
            weigths_init = w_init)
        # Scale output to action_bound to action_bound
        # NOTE: THIS IS DIFFERENT
        scale_out = tf.multiply(out,
                                self.action_bound)
        return inputs, out, scaled_out


    
