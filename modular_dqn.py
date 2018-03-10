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
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = None

    def build(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.model = model

    def fit(self, state, target_prediction):
        return self.model.fit(state, target_prediction, epochs=1, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class QModel:
    def __init__(self, model, target_model=None, tau=None, experience_replay=None, use_double_q=False):
        self.model = model
        self.target_model = target_model
        self.tau = tau
        self.experience_replay = experience_replay
        self.use_double_q = use_double_q

    def fit(self, state, target_prediction):
        if self.use_double_q and np.random.rand() < 0.5:
            swap = self.model
            self.model = self.target_model
            self.target_model = swap

        result = self.model.fit(state, target_prediction)
        return result

    def predict(self, state):
        return self.model.predict(state)

    def get_target_value(self, state):
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

    def update_target_values(self):
        #
        # Vanilla DQN - do nothing
        #
        if self.target_model is None:
            return

        #
        # Fixed-target DQN and/or Double-DQN
        #

        # TODO figure out how to get the slow-shifting tau weight from Silver 2016 to work correctly
        # weights_model = self.model.get_weights()
        # weights_target = self.target_model.get_weights()
        #
        # for i in range(len(weights_model)):
        #     weights_model[i] *= self.tau
        #
        # for i in range(len(weights_target)):
        #     weights_target[i] *= (1. - self.tau)
        #
        # self.target_model.set_weights(weights_model + weights_target)
        self.target_model.set_weights(self.model.get_weights())


class QAgent:
    def __init__(self,
                 state_size, action_size, model, exploration, discount_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.exploration = exploration
        self.gamma = discount_rate

    def act(self, state, be_greedy=False):
        if not be_greedy and self.exploration.should_explore():
            return random.randrange(self.action_size)

        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.model.experience_replay.add(state, action, reward, (next_state, done))

    def replay(self):
        if not self.model.experience_replay.can_sample():
            return

        minibatch = self.model.experience_replay.sample()

        for state, action, reward, (next_state, done) in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.model.get_target_value(next_state)

            target_prediction = self.model.predict(state)
            target_prediction[0][action] = target
            self.model.fit(state, target_prediction)

        self.exploration.step()

    def train(self, env, episode_length=1):
        state = env.reset()
        time_steps = 0

        for time_steps in range(episode_length):
            action = self.act(state)
            next_state, reward, done = env.step(action)
            self.model.experience_replay.add(state, action, reward, (next_state, done))
            state = next_state

            if done:
                self.model.update_target_values()
                break

        if self.model.experience_replay.can_sample():
            self.replay()

        return time_steps


class CartPoleEnvironmentWrapper:
    """
    A helper wrapper around the environment, to simplify agent code related to the reward.
    """

    def __init__(self, env):
        """
        Creates the wrapped environment.
        :param env: The environment to wrap.
        """
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

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
    # qmodel = FixedTargetQModel(model=model, target_model=target_model, tau=0.001, experience_replay=experience_replay)
    qmodel = QModel(model=model, target_model=target_model, tau=0.001,
                    experience_replay=experience_replay, use_double_q=True)

    agent = QAgent(state_size=env.state_size, action_size=env.action_size, model=qmodel, exploration=exploration,
                   discount_rate=0.95)

    scores = np.empty((num_episodes,))
    time_start = time.time()

    for e in range(num_episodes):
        time_steps = agent.train(env=env, episode_length=num_time_steps)
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e + 1, num_episodes, time_steps, agent.exploration.epsilon))
        scores[e] = time_steps

    time_end = time.time()
    print('Average last 1000 episodes:', np.mean(scores[1000:]))
    print('Time taken:', time_end - time_start, 'seconds')

    if scores_filename is not None:
        np.savetxt(scores_filename, scores, delimiter=',')


if __name__ == "__main__":
    run(CartPoleEnvironmentWrapper(gym.make('CartPole-v1')),
        num_episodes=10000, num_time_steps=500, replay_batch_size=32,
        scores_filename='modular_dqn.csv')
