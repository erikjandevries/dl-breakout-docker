# from keras.callbacks import CSVLogger
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import ProgbarLogger
# from keras.callbacks import TensorBoard
from keras.layers import Input, Lambda, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.layers.merge import Multiply
from keras.models import Model
from keras.optimizers import RMSprop
import logging
import numpy as np

from memory import Memory

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
fh = logging.FileHandler(str('./data/model_training.log'))
fh.setFormatter(formatter)
log.addHandler(ch)
log.addHandler(fh)

ATARI_SHAPE = (105, 80, 4)


class Agent:
    def __init__(self, env, memory_size=None):
        """
        Agent acting in an environment
        :param env:             The OpenAI Gym environment in which the Agent operates.
        :param memory_size:     The number of frames to keep in memory during training
        """

        self.env = env
        if memory_size is None:
            memory_size = 4
        self.memory = Memory(size=memory_size)
        self.model = self._build_model()

        self.callbacks = [
            # CSVLogger(filename='./data/log.csv', separator=',', append=True),
            # ModelCheckpoint(filepath='./data/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            # ModelCheckpoint(filepath='./data/weights.hdf5',
            #                 monitor='val_loss', verbose=0, save_best_only=False,
            #                 save_weights_only=False, mode='auto', period=10),
            # ProgbarLogger(count_mode='samples', stateful_metrics=None),
            # TensorBoard(log_dir='./data/logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
            #             write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        ]

        self._ba = 0

    def pick_random_action(self):
        return self.env.action_space.sample()

    def pick_best_action(self):
        state = np.stack([self.memory.get_state()], axis=0)
        actions = np.ones((1, self.env.action_space.n))
        q_values = self.model.predict([state, actions])
        best_action = np.argmax(q_values)
        # print("Best action: {}".format(best_action))
        self._ba += 1
        return best_action

    def train_step(self, gamma=0.99):
        states, a, next_states, rewards, dones, infos = self.memory.get_batch()

        actions = np.zeros((len(a), self.env.action_space.n))
        actions[np.arange(len(a)), a] = 1

        # print(next_states.shape, actions.shape, end="")

        next_q_values = self.model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_q_values[dones] = 0

        q_values = rewards + gamma * np.max(next_q_values, axis=1)
        targets = actions * q_values[:, None]

        # print(" - ", states.shape, targets.shape, end="")

        self.model.fit(
            [states, actions],
            targets,
            epochs=1,
            batch_size=len(states),
            verbose=0,
            callbacks=self.callbacks
        )

        # print("")

    def _build_model(self):
        return self._build_model_2015()

    def _build_model_2013(self):
        log.info("Building 2015 model")
        n_actions = self.env.action_space.n

        # With the functional API we need to define the inputs.
        frames_input = Input(ATARI_SHAPE, name='frames')
        actions_input = Input((n_actions,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = Lambda(lambda x: x / 255.0)(frames_input)

        # "The first hidden layer convolves 16 8x8 filters with stride 4 with the input image
        # and applies a rectifier nonlinearity."
        conv_1 = Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(normalized)

        # "The second hidden layer convolves 32 4x4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2 = Conv2D(32, (4, 4), activation="relu", strides=(2, 2))(conv_1)

        # Flattening the second convolutional layer.
        conv_flattened = Flatten()(conv_2)

        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = Dense(256, activation='relu')(conv_flattened)

        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = Dense(n_actions)(hidden)

        # Finally, we multiply the output by the mask!
        filtered_output = Multiply()([output, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')

        return model

    def _build_model_2015(self):
        log.info("Building 2015 model")
        n_actions = self.env.action_space.n

        # With the functional API we need to define the inputs.
        frames_input = Input(ATARI_SHAPE, name='frames')
        actions_input = Input((n_actions,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = Lambda(lambda x: x / 255.0)(frames_input)

        # "The first hidden layer convolves 32 8x8 filters with stride 4 with the input image
        # and applies a rectifier nonlinearity."
        conv_1 = Conv2D(32, (8, 8), activation="relu", strides=(4, 4))(normalized)

        # "The second hidden layer convolves 64 4x4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2 = Conv2D(64, (4, 4), activation="relu", strides=(2, 2))(conv_1)

        # "The third hidden layer convolves 64 3x3 filters with stride 1, again followed by a rectifier nonlinearity."
        conv_3 = Conv2D(64, (3, 3), activation="relu", strides=(1, 1))(conv_2)

        # Flattening the second convolutional layer.
        conv_flattened = Flatten()(conv_3)

        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = Dense(512, activation='relu')(conv_flattened)

        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = Dense(n_actions)(hidden)

        # Finally, we multiply the output by the mask!
        filtered_output = Multiply()([output, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')

        return model

    def load_model(self, filepath):
        self.model.load_weights(filepath)

    def save_model(self, filepath):
        self.model.save_weights(filepath)
