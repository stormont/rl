
from third_party.openai.baselines.baselines.deepq.replay_buffer import PrioritizedReplayBuffer


class PrioritizedExperience:
    """
    A behavior for storing prioritized experience replay storage and sampling.
    """

    @staticmethod
    def supports_prioritization():
        """
        Whether the class supports prioritized experience replay.
        :return: This class always supports prioritized experience replay.
        """
        return True

    def __init__(self, max_size, batch_size, replay_start_size, initial_td_error, epsilon, alpha, beta, anneal_rate):
        """
        Creates the behavior.
        :param max_size: The maximum storage size of the experience replay.
        :param batch_size: The number of experiences to sample per replay.
        :param replay_start_size: The required size of stored experience before experience can be replayed.
        :param initial_td_error: The initial TD-error to assign to prioritized samples.
        :param epsilon: The epsilon non-zero error to add to prioritized samples.
        :param alpha: The exponent by which to weight the prioritization amount.
        :param beta: The exponent by which to weight importance sampling.
        :param anneal_rate: The rate by which to anneal `alpha` and `beta` to 1.
        """
        self.memory = PrioritizedReplayBuffer(size=max_size, alpha=alpha)
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.initial_td_error = initial_td_error
        self.epsilon = epsilon
        self.beta = beta
        self.anneal_rate = anneal_rate

    def add(self, state, action, reward, next_state):
        """
        Adds an experience to the replay memory. Include all necessary environment state in the 'state'
        or 'next_state' variable (e.g., whether the episode has completed, for episodic environments).
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward provided for the (state, action) pair.
        :param next_state: The next state transitioned to.
        """
        self.memory.add(obs_t=state, action=action, reward=reward, obs_tp1=next_state[0], done=next_state[1])

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

        states, actions, rewards, next_states, dones, importances, indices = \
            self.memory.sample(batch_size=self.batch_size, beta=self.beta)
        samples = []

        # Map to shape: (state, action, reward, (next_state, done)), importance, index
        for i in range(len(states)):
            value = (states[i], actions[i], rewards[i], (next_states[i], dones[i]))
            samples.append((value, importances[i], indices[i]))

        return samples

    def step(self):
        """
        Indicate that a training step is complete to anneal `alpha` and `beta` towards one.
        """
        self.memory.anneal_alpha(self.anneal_rate)
        self.beta = max(self.beta * self.anneal_rate, 1.)

    def update_priority(self, sample_index, new_priority):
        """
        Updates the priority of the sample.
        :param sample_index: The index of the sample to update.
        :param new_priority: The new priority of the sample.
        """
        self.memory.update_priorities(idxes=[sample_index], priorities=[abs(new_priority) + self.epsilon])
