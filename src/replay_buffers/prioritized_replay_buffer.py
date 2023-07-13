import math

import torch
from gfn.envs import Env
from gfn.containers.trajectories import Trajectories


class PrioritizedReplayBuffer:
    def __init__(
            self,
            env: Env,
            capacity: int = 1000,
            fraction_of_samples_from_top=0.5,
            top_and_bottom_fraction=0.1
    ):
        self.env = env
        self.capacity = capacity
        self.terminating_states = env.States.from_batch_shape((0,))
        self.terminating_states.log_rewards = torch.empty(size=(0,)).to(env.device)
        self.fraction_of_samples_from_top = fraction_of_samples_from_top
        self.top_and_bottom_fraction = top_and_bottom_fraction

    def __len__(self):
        return self.terminating_states.batch_shape[0]

    def add(self, training_objects: Trajectories):
        all_states = torch.cat([
            self.terminating_states.states_tensor,
            training_objects.last_states.states_tensor
        ])
        all_log_rewards = torch.cat([
            self.terminating_states.log_rewards,
            training_objects.log_rewards
        ])

        unique_indices = self._get_indices_of_unique_states(all_states)
        unique_states = all_states[unique_indices]
        unique_log_rewards = all_log_rewards[unique_indices]

        sorted_log_rewards, sorted_inds = torch.sort(unique_log_rewards, descending=True, stable=True)
        sorted_states = unique_states[sorted_inds]

        n_states = sorted_states.shape[0]
        if n_states > self.capacity:
            half = math.ceil(self.capacity / 2)
            other_half = self.capacity - half
            sorted_states = torch.cat([sorted_states[:half], sorted_states[-other_half:]])
            sorted_log_rewards = torch.cat([sorted_log_rewards[:half], sorted_log_rewards[-other_half:]])

        self.terminating_states = self.env.make_States_class()(sorted_states)
        self.terminating_states.log_rewards = sorted_log_rewards

    def sample(self, n_samples):
        get_n_samples_from_top = math.ceil(self.fraction_of_samples_from_top * n_samples)
        get_n_samples_from_bottom = n_samples - get_n_samples_from_top

        n_stored = self.terminating_states.batch_shape[0]
        choose_from_n = math.ceil(self.top_and_bottom_fraction * n_stored)

        top_states = self.terminating_states[:choose_from_n].sample(get_n_samples_from_top)
        bottom_states = self.terminating_states[-choose_from_n:].sample(get_n_samples_from_bottom)

        sampled_states = self.env.States.from_batch_shape((0,))
        sampled_states.extend(top_states)
        sampled_states.extend(bottom_states)
        return sampled_states


    # https://github.com/pytorch/pytorch/issues/36748#issuecomment-1478913448
    @staticmethod
    def _get_indices_of_unique_states(x, dim=0):
        unique, inverse, counts = torch.unique(
            x, dim=dim, sorted=True, return_inverse=True, return_counts=True
        )
        inv_sorted = inverse.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        unique_indices = inv_sorted[tot_counts]
        return unique_indices

