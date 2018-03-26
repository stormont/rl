
from egreedy import EpsilonGreedy
from experience import Experience
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym
import numpy as np
# from prioritized_experience import PrioritizedExperience
from qlearning import QAgent, QModel
import time


class ExampleModel:
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
    experience_replay = Experience(
        max_size=experience_max_size, batch_size=replay_batch_size, replay_start_size=replay_start_size)
    # experience_replay = PrioritizedExperience(
    #     max_size=experience_max_size, batch_size=replay_batch_size, replay_start_size=replay_start_size,
    #     initial_td_error=10, alpha=0.4, beta=0.4, anneal_rate=0.95, epsilon=0.001)

    model = ExampleModel(state_size=env.state_size, action_size=env.action_size, learning_rate=0.001)
    model.build()
    target_model = ExampleModel(state_size=env.state_size, action_size=env.action_size, learning_rate=0.001)
    target_model.build()

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
