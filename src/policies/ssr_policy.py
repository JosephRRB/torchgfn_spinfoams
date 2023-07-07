from abc import abstractmethod

import torch
from torch.nn.functional import one_hot

from gfn.estimators import FunctionEstimator
from gfn.envs import Env
from gfn.containers import States
from gfn.modules import NeuralNet


class BaseSSRPolicy(FunctionEstimator):
    def __init__(
        self, env: Env, **nn_kwargs,
    ) -> None:
        self.env = env
        self.module = NeuralNet(
            input_dim=2 * self.env.grid_dim * self.env.grid_len,
            output_dim=1,
            **nn_kwargs
        )

    @staticmethod
    def _encode_possible_actions(masks):
        _, possible_actions = torch.where(masks)
        encoded_possible_actions = one_hot(possible_actions)
        return encoded_possible_actions

    @staticmethod
    def _repeat_source_states(source_states, masks):
        n_destinations = torch.sum(masks, dim=1)
        duplicated_source_states = torch.repeat_interleave(
            source_states, n_destinations, dim=0
        )
        return duplicated_source_states

    @staticmethod
    @abstractmethod
    def _get_destination_states(
            duplicated_source_states, encoded_possible_actions
    ):
        pass

    def _encode_states(self, states):
        encoded_states = torch.flatten(
            one_hot(states, num_classes=self.env.grid_len).float(),
            start_dim=1
        )
        return encoded_states

    @abstractmethod
    def _encode_source_destination_state_pairs(self, states, masks):
        pass

    @staticmethod
    def _organize_logits(logits, masks):
        masked_logits = torch.full_like(
            masks, fill_value=-float("inf"), dtype=torch.float
        )
        masked_logits[masks] = logits.ravel()
        return masked_logits

    def _get_masked_logits(self, states, masks):
        encoded_pairs = self._encode_source_destination_state_pairs(states, masks)
        logits = self.module(encoded_pairs)
        masked_logits = self._organize_logits(logits, masks)
        return masked_logits


class ForwardLogRelativeEdgeFlowEstimator(BaseSSRPolicy):
    def __init__(
        self, env: Env, **nn_kwargs,
    ) -> None:
        super().__init__(env, **nn_kwargs)

    def __call__(self, states: States):
        return self._get_masked_logits(states.states_tensor, states.forward_masks)

    @staticmethod
    def _get_destination_states(
        duplicated_source_states, encoded_possible_actions
    ):
        destination_states = duplicated_source_states + encoded_possible_actions
        return destination_states

    def _encode_source_destination_state_pairs(self, states, masks):
        encoded_forward_actions = self._encode_possible_actions(masks)[:, :-1]
        duplicated_states = self._repeat_source_states(states, masks)
        children = self._get_destination_states(duplicated_states, encoded_forward_actions)

        non_sf_children_mask = torch.sum(encoded_forward_actions, dim=1).bool()
        encoded_children = torch.zeros(
            size=(children.shape[0], self.env.grid_dim * self.env.grid_len)
        )
        encoded_children[non_sf_children_mask] = self._encode_states(children[non_sf_children_mask])

        encoded_states = self._encode_states(duplicated_states)
        encoded_pairs = torch.cat([
            encoded_states, encoded_children
        ], dim=1)
        return encoded_pairs


class BackwardLogRelativeEdgeFlowEstimator(BaseSSRPolicy):
    def __init__(
        self, env: Env, **nn_kwargs,
    ) -> None:
        super().__init__(env, **nn_kwargs)

    def __call__(self, states: States):
        return self._get_masked_logits(states.states_tensor, states.backward_masks)

    @staticmethod
    def _get_destination_states(
            duplicated_source_states, encoded_possible_actions
    ):
        destination_states = duplicated_source_states - encoded_possible_actions
        return destination_states

    def _encode_source_destination_state_pairs(self, states, masks):
        encoded_backward_actions = self._encode_possible_actions(masks)
        duplicated_states = self._repeat_source_states(states, masks)
        parents = self._get_destination_states(duplicated_states, encoded_backward_actions)
        encoded_parents = self._encode_states(parents)
        encoded_states = self._encode_states(duplicated_states)
        encoded_pairs = torch.cat([
            encoded_states, encoded_parents
        ], dim=1)
        return encoded_pairs





