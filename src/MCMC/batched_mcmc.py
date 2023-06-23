import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
from scipy.stats import norm


class MCMCRunner:
    def __init__(
            self,
            grid_rewards: np.ndarray,
            proposal_distribution_scale=0.8,
    ):
        side_len = np.unique(grid_rewards.shape)
        assert len(side_len) == 1
        self.grid_length = side_len[0]
        self.grid_dim = len(grid_rewards.shape) # grid_rewards.ndim
        self.scale = proposal_distribution_scale

        self.grid_rewards = grid_rewards

        self.rand_gen = np.random.default_rng()
        self.grid_coords_1d = np.arange(self.grid_length)
        self.proposal_distributions_1d = self._calculate_next_1d_coordinate_distributions()



    @staticmethod
    def _discrete_normal_distribution(coords, center, scale=1.0):
        return norm.cdf(coords + 0.5, loc=center, scale=scale) - norm.cdf(coords - 0.5, loc=center, scale=scale)

    def _calculate_next_1d_coordinate_distributions(self):
        distributions_1d = self._discrete_normal_distribution(
            self.grid_coords_1d, self.grid_coords_1d.reshape(-1, 1), scale=self.scale
        )
        distributions_1d /= np.sum(distributions_1d, axis=1, keepdims=True)
        return distributions_1d

    def generate_initial_coordinates(self, batch_size):
        coords = self.rand_gen.integers(self.grid_length, size=(batch_size, self.grid_dim))
        return coords

    def _choose_next_1d_coordinates(self, current_1d_coordinate, choose_n):
        next_1d_coords = self.rand_gen.choice(
            self.grid_coords_1d, p=self.proposal_distributions_1d[current_1d_coordinate, :], size=choose_n
        )
        return next_1d_coords

    def choose_next_coordinates(self, current_coordinates):
        coords_1d = current_coordinates.flatten()
        unique_coords_1d = np.unique(coords_1d)
        inds_of_coords = [np.where(coords_1d == unique)[0] for unique in unique_coords_1d]
        for unique, ind in zip(unique_coords_1d, inds_of_coords):
            coords_1d[ind] = self._choose_next_1d_coordinates(unique, len(ind))

        next_coordinates = coords_1d.reshape(-1, self.grid_dim)
        return next_coordinates

    def _get_next_probabilities_1d(self, current_1d_coordinates, next_1d_coordinates):
        next_probas_1d = self.proposal_distributions_1d[current_1d_coordinates, next_1d_coordinates]
        return next_probas_1d

    def _get_next_probabilities(self, current_coordinates, next_coordinates):
        next_probas_1d = self._get_next_probabilities_1d(
            current_coordinates.ravel(), next_coordinates.ravel()
        )
        next_probas = np.product(next_probas_1d.reshape(-1, self.grid_dim), axis=1)
        return next_probas

    def _get_grid_rewards(self, grid_coordinates):
        inds = [grid_coordinates[:, i] for i in range(self.grid_dim)]
        rewards = self.grid_rewards[*inds]
        return rewards

    def _calculate_acceptance_probabilities(self, current_coordinates, next_coordinates):
        forward_proposal_probas = self._get_next_probabilities(current_coordinates, next_coordinates)
        backward_proposal_probas = self._get_next_probabilities(next_coordinates, current_coordinates)

        current_probas = self._get_grid_rewards(current_coordinates)
        next_probas = self._get_grid_rewards(next_coordinates)

        forward = current_probas * forward_proposal_probas
        backward = next_probas * backward_proposal_probas

        acceptance_probabilities = np.minimum(1.0, backward / forward)
        return acceptance_probabilities

    def run_mcmc_chains(
            self, batch_size, n_iterations, generated_data_dir
    ):
        os.makedirs(generated_data_dir, exist_ok=True)
        current_coordinates = self.generate_initial_coordinates(batch_size)

        mcmc_chains = []
        acceptance_masks = []
        for _ in tqdm(range(n_iterations)):
            next_coordinates = self.choose_next_coordinates(current_coordinates)
            acceptance_probabilities = self._calculate_acceptance_probabilities(
                current_coordinates, next_coordinates
            )
            uniform_random = self.rand_gen.random(size=batch_size)
            acceptance_mask = uniform_random <= acceptance_probabilities
            current_coordinates[acceptance_mask] = next_coordinates[acceptance_mask]

            mcmc_chains.append(current_coordinates.copy())
            acceptance_masks.append(acceptance_mask.copy())

        mcmc_chains = np.stack(mcmc_chains)
        acceptance_masks = np.stack(acceptance_masks)

        np.save(
            f"{generated_data_dir}/mcmc_chains.npy",
            mcmc_chains
        )
        np.save(
            f"{generated_data_dir}/acceptance_masks.npy",
            acceptance_masks
        )
        return mcmc_chains, acceptance_masks
