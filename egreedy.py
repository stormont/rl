
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
