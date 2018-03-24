
import numpy as np
import random


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
        :return: The total reward for the training episode.
        """
        state = env.reset()
        total_reward = 0

        for steps_taken in range(episode_length):
            action = self.act(state)
            next_state, reward, done = env.step(action)
            self.model.experience_replay.add(state, action, reward, (next_state, done))
            state = next_state
            total_reward += reward

            if done:
                break

        if self.model.experience_replay.can_sample():
            self._replay()

        return total_reward

    def _replay(self):
        """
        Performs a step of experience replay for additional offline training.
        """
        if not self.model.experience_replay.can_sample():
            return

        minibatch = self.model.experience_replay.sample()

        for sample in minibatch:
            if self.model.experience_replay.supports_prioritization():
                (state, action, reward, (next_state, done)), importance, index = sample
            else:
                state, action, reward, (next_state, done) = sample
                importance = 1
                index = None  # Placeholder to prevent unnecessary code analysis warning

            if done:
                td_error = reward
            else:
                td_error = reward + self.gamma * self.model.get_target_value(next_state)

            if self.model.experience_replay.supports_prioritization():
                self.model.experience_replay.update_priority(index, td_error)

            td_error = importance * td_error
            target_prediction = self.model.predict(state)
            target_prediction[0][action] = td_error
            self.model.fit(state, target_prediction)

            if self.model.supports_soft_target_updates():
                self.model.update_target_values()

        self.model.experience_replay.step()
        self.exploration.step()
