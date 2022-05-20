import numpy as np


class Player:
    """
    Base class for both types of players (QLearning, DQN).
    """

    def __init__(self, player='X', epsilon=0.05, seed=666):
        self.player = player
        self.epsilon = epsilon

        # RNG for the epsilon-gredy policy
        self.rng_ = np.random.default_rng(seed=seed)

    def act(self, grid):
        """
        Selects an action to perform based on the current
        grid state and the player's policy.
        """
        raise NotImplementedError("Call from abstract class")

    def set_player(self, player='X', j=-1):
        self.player = player
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    @staticmethod
    def empty(grid):
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i / 3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def randomMove(self, grid):
        """ Chose a random move from the available options. """
        avail = self.empty(grid)

        return avail[self.rng_.integers(0, len(avail))]