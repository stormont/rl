from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym
import numpy as np
import random
import time


class Experience:
    """
    A behavior for defining experience replay storage and sampling.
    """

    def __init__(self, max_size, batch_size, replay_start_size):
        """
        Creates the behavior.
        :param max_size: The maximum storage size of the experience replay.
        :param batch_size: The number of experiences to sample per replay.
        :param replay_start_size: The required size of stored experience before experience can be replayed.
        """
        self.memory = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

    def add(self, state, action, reward, next_state):
        """
        Adds an experience to the replay memory. Include all necessary environment state in the 'state'
        or 'next_state' variable (e.g., whether the episode has completed, for episodic environments).
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward provided for the (state, action) pair.
        :param next_state: The next state transitioned to.
        """
        self.memory.append((state, action, reward, next_state))

    def can_sample(self):
        """
        Checks whether experience replay is allowed to sample.
        :return: Whether experience replay is allowed to sample batches.
        """
        return len(self.memory) >= self.replay_start_size

    def sample(self):
        """
        Samples randomly from the stored experience.
        :return: A batch of random samples from the stored experience, or an empty array if not allowed.
        """
        if not self.can_sample():
            return []

        return random.sample(self.memory, self.batch_size)


class EpsilonGreedy:
    """
    A behavior for managing exploration, using the epsilon-greedy algorithm.
    """

    def __init__(self, epsilon_start, epsilon_min, epsilon_decay):
        """
        Creates the behavior.
        :param epsilon_start: The start value for epsilon.
        :param epsilon_min: The minimum value for epsilon.
        :param epsilon_decay: The rate at which epsilon should be decayed.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def should_explore(self):
        """
        Checks whether the behavior indicates to explore or exploit (be greedy).
        :return: Whether to explore (False indicates to exploit/be greedy).
        """
        return np.random.rand() <= self.epsilon

    def step(self):
        """
        Indicate that an epoch step has completed. This decays epsilon.
        :return: None
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class Model:
    """
    A generic dense neural network model.
    """
    def __init__(self, state_size, action_size, learning_rate):
        """
        Creates the model

        :param state_size: The size of the state space.
        :param action_size: The size of the action space.
        :param learning_rate: The learning rate to apply during training.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = None

    def build(self):
        """
        Builds the model.
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.model = model

    def fit(self, state, target_prediction):
        """
        Fits a state and target prediction against the model (with one epoch).

        :param state: The state to fit.
        :param target_prediction: The target prediction.
        """
        self.model.fit(state, target_prediction, epochs=1, verbose=0)

    def predict(self, state):
        """
        Get a prediction for the state from the model.

        :param state: The state to get a prediction for.
        :return: The predicted outcome.
        """
        return self.model.predict(state)

    def get_weights(self):
        """
        Gets the current weights for the model.

        :return: The current weights.
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        """
        Sets the current weights for the model.

        :param weights: The weights to set.
        """
        self.model.set_weights(weights)

    def load(self, filename):
        """
        Loads weights from a file into the model.

        :param filename: The name of the file that contains the weight data.
        """
        self.model.load_weights(filename)

    def save(self, filename):
        """
        Saves model weights to a file.

        :param filename: The name of the file to save the weight data to.
        """
        self.model.save_weights(filename)


class QModel:
    """
    A model for performing Q-learning.
    """

    def __init__(self, model, target_model=None, tau=None, experience_replay=None, use_double_q=False):
        """
        Creates the Q-learning model.

        :param model: The model for current action and training.
        :param target_model: (optional) The target model for training against.
        :param tau: (optional) The weight to apply for updating target weights to the current model. When this is not
                    supplied, a strict copy will be used, as per Mnih 2015.
        :param experience_replay: (optional) The behavior for performing experience replay.
        :param use_double_q: Whether to use Double-Q learning.
        """
        self.model = model
        self.target_model = target_model
        self.tau = tau
        self.experience_replay = experience_replay
        self.use_double_q = use_double_q

    def fit(self, state, target_prediction):
        """
        Fits a state and target prediction against the model.

        :param state: The state to fit.
        :param target_prediction: The target prediction.
        """
        if self.use_double_q and np.random.rand() < 0.5:
            swap = self.model
            self.model = self.target_model
            self.target_model = swap

        self.model.fit(state, target_prediction)

    def predict(self, state):
        return self.model.predict(state)

    def get_target_value(self, state):
        """
        Gets the TD-target for the given state.

        :param state: The state to get the TD-target for.
        :return: The TD-target of the state.
        """
        #
        # Vanilla DQN
        #
        if self.target_model is None:
            # Equivalent representations:
            #   Yt_Q ≡ Rt+1 + γ max_a Q(St+1, a; θt)
            #   Yt_Q ≡ Rt+1 + γ Q(St+1, argmax_a Q(St+1, a; θt), θt)
            return np.amax(self.model.predict(state)[0])

        #
        # Fixed-target DQN and/or Double-DQN
        #

        # Equivalent representations:
        #   Yt_Q ≡ Rt+1 + γ max_a Q(St+1, a; θt-)
        #   Yt_Q ≡ Rt+1 + γ Q(St+1, argmax_a Q(St+1, a; θt), θt-)
        #   Yt_DoubleDQN ≡ Rt+1 + γ Q(St+1, argmax_a Q(St+1, a; θt), θt')
        action = np.argmax(self.model.predict(state)[0])
        target_values = self.target_model.predict(state)[0]
        return target_values[action]

    def supports_soft_target_updates(self):
        """
        Gets whether the model supports frequent, "soft" gradual updates to target values.

        :return: Whether soft target updates are supported.
        """
        return self.tau is not None

    def update_target_values(self):
        """
        Updates the target values for fixed target DQN or Double-Q learning. (This is a no-op when using vanilla DQN.)
        """
        #
        # Vanilla DQN - do nothing
        #
        if self.target_model is None:
            return

        #
        # Fixed-target DQN and/or Double-DQN
        #

        if not self.supports_soft_target_updates():
            # Just do a hard copy, as per [Mnih 2015]
            self.target_model.set_weights(self.model.get_weights())
            return

        # Otherwise, use soft target updates, as per [Lillicrap 2016]
        weights_model = self.model.get_weights()
        weights_target = self.target_model.get_weights()
        new_weights = []

        for i in range(len(weights_model)):
            new_weights.append(self.tau * weights_model[i] + (1. - self.tau) * weights_target[i])

        self.target_model.set_weights(new_weights)


class QAgent:
    """
    An agent for exploring and taking action in an environment with a discrete action space.
    """

    def __init__(self, state_size, action_size, model, exploration, discount_rate):
        """
        Creates the agent.

        :param state_size: The size of the state space.
        :param action_size: The size of the discrete action space.
        :param model: The model to train and predict upon.
        :param exploration: The behavior for exploration.
        :param discount_rate: The discount rate for future rewards.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.exploration = exploration
        self.gamma = discount_rate

    def act(self, state, be_greedy=False):
        """
        Select an action to perform an action upon the environment.

        :param state: The state to act upon.
        :param be_greedy: Whether to act greedily.
        :return: The action to be performed on the environment.
        """
        if not be_greedy and self.exploration.should_explore():
            return random.randrange(self.action_size)

        return np.argmax(self.model.predict(state)[0])

    def train(self, env, episode_length=1):
        """
        Performs a training step for the model against the given environment.

        :param env: The environment to train against.
        :param episode_length: The max length of the episode to train against.
        :return: The number of steps taken in the episode.
        """
        state = env.reset()
        steps_taken = 0

        for steps_taken in range(episode_length):
            action = self.act(state)
            next_state, reward, done = env.step(action)
            self.model.experience_replay.add(state, action, reward, (next_state, done))
            state = next_state

            if done:
                break

        if self.model.experience_replay.can_sample():
            self._replay()

        return steps_taken

    def _replay(self):
        """
        Performs a step of experience replay for additional offline training.
        """
        if not self.model.experience_replay.can_sample():
            return

        minibatch = self.model.experience_replay.sample()

        for state, action, reward, (next_state, done) in minibatch:
            if done:
                td_error = reward
            else:
                td_error = reward + self.gamma * self.model.get_target_value(next_state)

            target_prediction = self.model.predict(state)
            target_prediction[0][action] = td_error
            self.model.fit(state, target_prediction)

            if self.model.supports_soft_target_updates():
                self.model.update_target_values()

        self.exploration.step()


class WrappedCartPoleEnvironment:
    """
    A helper wrapper around the CartPole environment, to simplify agent code related to the reward.
    """

    def __init__(self):
        """
        Creates the wrapped environment.
        """
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

    def reset(self):
        """
        Resets the environment to the starting state.
        :return: The starting state.
        """
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        return state

    def step(self, action):
        """
        Take a step in the environment.
        :param action: The action to take to move the environment forward.
        :return: The next state, provided reward, and whether the environment has ended.
        """
        next_state, reward, done, _ = self.env.step(action)
        # Discourage running out of time by adding a penalty
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, self.state_size])
        return next_state, reward, done


def run(env, num_episodes, num_time_steps, replay_batch_size, scores_filename=None):
    exploration = EpsilonGreedy(epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999)

    # [Mnih 2015] used:
    #  - replay over 2% of the total experience
    #  - batch size of 32
    #  - minimum replay start size of 0.1%
    experience_max_size = int(num_episodes * num_time_steps * 0.02)
    replay_start_size = int(num_episodes * num_time_steps * 0.001)
    experience_replay = Experience(max_size=experience_max_size, batch_size=replay_batch_size,
                                   replay_start_size=replay_start_size)

    model = Model(state_size=env.state_size, action_size=env.action_size, learning_rate=0.001)
    model.build()
    target_model = Model(state_size=env.state_size, action_size=env.action_size, learning_rate=0.001)
    target_model.build()

    # qmodel = QModel(model=model, experience_replay=experience_replay)
    # qmodel = FixedTargetQModel(model=model, target_model=target_model, experience_replay=experience_replay)
    qmodel = QModel(model=model, target_model=target_model, experience_replay=experience_replay,
                    tau=0.1, use_double_q=True)

    agent = QAgent(state_size=env.state_size, action_size=env.action_size, model=qmodel, exploration=exploration,
                   discount_rate=0.95)

    scores = np.empty((num_episodes,))
    time_start = time.time()

    for e in range(num_episodes):
        scores[e] = agent.train(env=env, episode_length=num_time_steps)
        print('episode: {}/{}, score: {}, e: {:.2}'.format(e + 1, num_episodes, scores[e], agent.exploration.epsilon))

    time_end = time.time()
    print('Average score for last 10% of episodes:', np.mean(scores[int(np.floor(num_episodes * 0.1)):]))
    print('Time taken:', time_end - time_start, 'seconds')

    if scores_filename is not None:
        np.savetxt(scores_filename, scores, delimiter=',')


if __name__ == "__main__":
    run(WrappedCartPoleEnvironment(),
        num_episodes=10000, num_time_steps=500, replay_batch_size=32)