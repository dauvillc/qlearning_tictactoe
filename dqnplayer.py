import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from random import sample
from player import Player


class DQN(nn.Module):
    """
    Neural network used by the DQN Agent.
    state (3,3,2)->flatten->(18,) nn input
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.flatten = nn.Flatten()
        self.input = nn.Linear(18, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 9)

    def forward(self, x):
        x = x.to(self.device)
        # Flattens x to make sure it can be passed to the linear layers
        x = self.flatten(x)
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # No activation is a linear activation
        return self.output(x)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """
    Replay memory used by the DQN Agent. Taken from the PyTorch DQN Tutorial:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    The memory is circular: when its max capacity is overcome, new data replaces
    the former experiences.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNPlayer(Player):
    """
    Implements a type of Player that uses Deep Q-Learning
    to learn the tictactoe strategy.
    """

    def __init__(self, device, player='X', lr=5e-4, discount=0.99, epsilon=lambda _: 0.05, batch_size=64,
                 use_replay=True, use_agent_weights=None,
                 seed=666):
        super().__init__(epsilon, player, seed)
        self.lr = lr
        self.discount = discount
        self.device = device

        self.last_action, self.last_state = None, None

        # Neural networks
        self.uses_external_weights = use_agent_weights is not None
        if not self.uses_external_weights:
            self.policy_net = DQN(device).to(device)
            self.target_net = DQN(device).to(device)

            self.use_replay = use_replay
            if use_replay:
                self.memory = ReplayMemory(10000)
                self.batch_size = batch_size
            else:
                self.batch_size = 1

            # Huber loss
            self.criterion = nn.HuberLoss()
            # Adam optimizer
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                              lr=self.lr)
        else:
            if not use_replay:
                raise ValueError('A DQN agent must use the replay memory if it using external weights')
            self.memory = use_agent_weights.memory
            self.policy_net = use_agent_weights.policy_net
            self.target_net = use_agent_weights.target_net

    def learn(self, reward, grid):
        """
        Stores the last (S, A, NS, R) tuple into the replay memory,
        and trains the policy network using a sample from the replay memory.
        --reward: float, end game reward
        --grid: (3, 3, 2) array representing the current state.
        Returns the value of the Huber loss.
        """
        self.memory.push(self.last_state, self.last_action, grid, reward)
        # An agent that uses external weights does not actually learn, it only
        # adds its experience to the memory
        if self.uses_external_weights:
            return None
        if self.use_replay:
            # We don't start learning before the replay memory is large enough to
            # return at least a full batch
            if len(self.memory) < self.batch_size:
                return 0

            # ## Policy network training ==========================================
            # Taken from the Pytorch RL tutorial
            # First sample a batch of Transition objects
            transitions = self.memory.sample(self.batch_size)
            # Then creates a single Transition obj whose elements are arrays
            batch = Transition(*zip(*transitions))
        else:
            batch = Transition([self.last_state], [self.last_action],
                               [grid], [reward])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Computes the state-action values for all actions for all states in
        # the batch
        state_action_values = self.policy_net(state_batch)
        # For each state, selects only Q(s, a) for the a which was
        # actually chosen
        state_action_values = state_action_values.gather(1, action_batch)

        # We now need to compute max_a Q(s', a)
        next_state_qvalues = torch.zeros(self.batch_size, device=self.device)
        # Contains only the next states when they were not final
        non_final_next_states= [ns for ns in batch.next_state if ns is not None]
        if non_final_next_states:
            non_final_next_states = torch.cat(non_final_next_states)
            non_final_mask = [ns is not None for ns in batch.next_state]
            # We'll set it to zero for final states
            # Make sure to use the target network (not the policy) for
            # training stability.
            # Note that tensor.max(dim=...) returns a namedtuple (values, indices)
            next_state_qvalues[non_final_mask] = \
                self.target_net(non_final_next_states).max(dim=1).values
        # Detach the next state values from the gradient graph as it will be
        # used as the target in the computation of the loss (We consider it as
        # the "true qvalue" and hope to converge towards the Bellman equation).
        next_state_qvalues = next_state_qvalues.detach()

        # Final objective term
        target = reward_batch + self.discount * next_state_qvalues

        # Loss minimization using the optimizer (usual PyTorch training phase)
        loss = self.criterion(state_action_values, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Returns the loss as a float value
        return loss.item()

    def update(self):
        """
        Udpates the target network by setting its weights
        to those of the policy network.
        """
        state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(state_dict)

    def act(self, grid, iteration):
        """
        Chooses the action to perform by taking that which returns the
        best qvalue, as estimated by the policy network.
        Returns the action taken as an integer from 0 to 8.
        """
        # We don't want those computations to impact the gradient graph
        # somehow
        with torch.no_grad():
            # Check whether the epsilon-greedy choice activates
            if self.rng_.random() < self.epsilon(iteration):
                action = torch.tensor([[self.rng_.integers(0, 9)]], device=self.device)
            else:
                qvalues = self.policy_net(grid)
                # Select the action that has the highest qvalue
                # Note that tensor.max(dim=...) returns a namedtuple (values, indices)
                action = qvalues.max(dim=1).indices
                # The action must have shape (1, 1) so that they can be concatenated
                # when sampled from the replay memory
                action = action.view(1, 1)

            self.last_state = torch.clone(grid)
            self.last_action = action
            return int(action.item())
