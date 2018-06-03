from keras import layers, optimizers,models
from keras import backend as K

class Critic():
    """Critic model Q(s,a)"""

    def __init__(self,state_size,action_size):
        """Initialize model
        
        Params:
        =======
            state_size(int): dimension of observation space
            action_size(int): dimension of action space
        """

        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        """Return critic network model"""

        # Define inputs 
        states = layers.Input(shape=self.state_size,name='states')
        actions = layers.Input(shape=self.action_size,name='actions')

        # Define states hidden layers
        net_states = layers.Dense(units=32,activation='relu')(states)
        net_states = layers.Dense(units=64,activation='relu')(net_states)

        # Define action hidden layers
        net_actions = layers.Dense(units=32,activation='relu')(actions)
        net_actions = layers.Dense(units=64,activation='relu')(net_states)

        # Merge action and states sub-networks 
        net = layers.add([net_states,net_actions])
        net = layers.Activation('relu')(net)

        # Define output layer
        Q_values = layers.Dense(units=1,name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer,loss function and compile model
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer,loss='mse')

        # Compute action gradients
        action_gradients = K.gradients(Q_values,actions)

        # Function returning action gradients to actor network
        self.get_action_gradients = K.Function(
            inputs = [*self.model.input,K.learning_phase()],
            outputs = action_gradients
        )