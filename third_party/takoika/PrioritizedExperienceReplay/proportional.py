import random
import third_party.takoika.PrioritizedExperienceReplay.sum_tree as sum_tree


class Experience(object):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self, memory_size, alpha, epsilon):
        """ Prioritized experience replay buffer initialization.
        
        Parameters
        ----------
        memory_size : int
            sample size to be stored
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        epsilon: float
            small positive constant to prevent zero weight priorities
        """
        self.tree = sum_tree.SumTree(memory_size)
        self.memory_size = memory_size
        self.alpha = alpha
        self.epsilon = epsilon

    def __len__(self):
        return self.tree.filled_size()

    def add(self, data, priority):
        """ Add new sample.
        
        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, abs(priority)**self.alpha)

    def select(self, beta, batch_size=1):
        """ The method return samples randomly.
        
        Parameters
        ----------
        beta : float
        batch_size : int
            batch size to be selected
        
        Returns
        -------
        samples :
            list of samples
        weights: 
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """
        
        if self.tree.filled_size() < batch_size:
            return None, None, None

        samples = []
        indices = []
        weights = []
        priorities = []

        for _ in range(batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append(max((1./self.memory_size/priority)**beta, self.epsilon))
            indices.append(index)
            samples.append(data)
            self.priority_update([index], [0])  # To avoid duplicating

        self.priority_update(indices, priorities)  # Revert priorities
        weights /= max(weights)  # Normalize for stability
        return zip(samples, weights, indices)

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.
        
        Parameters
        ----------
        indices : 
            list of sample indices
        priorities :
            list of updated priorities
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, abs(p)**self.alpha)
    
    def set_alpha(self, alpha):
        """ Set the exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)
