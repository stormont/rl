
from collections import deque
import random


class Experience:
    """
    A behavior for storing experience replay storage and sampling.
    """

    @staticmethod
    def supports_prioritization():
        """
        Whether the class supports prioritized experience replay.
        :return: This class never supports prioritized experience replay.
        """
        return False

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
        value = (state, action, reward, next_state)

        self.memory.append(value)

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

    def step(self):
        """
        Indicate that a training step is complete.
        """
        pass
