from typing import ClassVar, Tuple, Literal, cast

import numpy as np
import torch
from torchtyping import TensorType


from gymnasium.spaces import Discrete
from gfn.envs.preprocessors import KHotPreprocessor
from gfn.envs.env import Env
from gfn.containers.states import States

import random


# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
BatchTensor = TensorType["batch_shape"]


class BaseGrid(Env):
    def __init__(
            self,
            grid_rewards: np.ndarray,
            device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        side_len = np.unique(grid_rewards.shape)
        assert len(side_len) == 1
        self.grid_len = side_len[0]
        self.grid_dim = len(grid_rewards.shape)
        self.device = torch.device(device_str)

        self.grid_rewards = torch.tensor(grid_rewards, dtype=torch.float).to(self.device)
        self.log_grid_rewards = torch.log(self.grid_rewards)

        s0 = torch.zeros(
            self.grid_dim, dtype=torch.long, device=self.device
        )
        
        #s0 = torch.randint(self.grid_len, (self.grid_dim,))
        
        sf = torch.full(
            (self.grid_dim,), fill_value=-1, dtype=torch.long, device=self.device
        )

        action_space = Discrete(self.grid_dim + 1)

        preprocessor = KHotPreprocessor(
            height=self.grid_len, ndim=self.grid_dim, get_states_indices=None
        )

        super().__init__(
            action_space=action_space,
            s0=s0,
            sf=sf,
            preprocessor=preprocessor,
        )

    def make_States_class(self) -> type[States]:
        "Creates a States class for this environment"
        env = self

        class GridStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.grid_dim,)
            s0 = env.s0
            sf = env.sf
            
            #@classmethod
            #def make_random_states_tensor(cls, batch_shape: tuple[int]) -> StatesTensor:
            #    assert cls.s0 is not None and state_ndim is not None
            #    return cls.s0.repeat(*batch_shape, * random.sample(range(self.grid_len), self.grid_dim) )

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                forward_masks = torch.ones(
                    (*self.batch_shape, env.n_actions),
                    dtype=torch.bool,
                    device=env.device,
                )
                backward_masks = torch.ones(
                    (*self.batch_shape, env.n_actions - 1),
                    dtype=torch.bool,
                    device=env.device,
                )

                return forward_masks, backward_masks

            def update_masks(self) -> None:
                "Update the masks based on the current states."
                # The following two lines are for typing only.
                self.forward_masks = cast(ForwardMasksTensor, self.forward_masks)
                self.backward_masks = cast(BackwardMasksTensor, self.backward_masks)

                self.forward_masks[..., :-1] = self.states_tensor != env.grid_len - 1
                self.backward_masks = self.states_tensor != 0

        return GridStates

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        return actions == self.action_space.n - 1
#        return torch.ones(actions.size(dim=0), dtype = torch.bool)

    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), 1, reduce="add")

    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), -1, reduce="add")

    def log_reward(self, final_states: States) -> TensorFloat:
        final_states_raw = final_states.states_tensor
        indices = [
            final_states_raw[:, i]
            for i in range(self.grid_dim)
        ]
        return self.log_grid_rewards[indices]

    @property
    def n_states(self) -> int:
        return self.grif_len**self.grid_dim

    @property
    def all_states(self) -> States:
        # This is brute force !
        digits = torch.arange(self.grid_len, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.grid_dim)
        return self.States(all_states)
    
   
    @property
    def n_terminating_states(self) -> int:
        digits = torch.arange(self.grid_len)
        all_states = torch.cartesian_prod(*[digits] * self.grid_dim)
        terminating_states = [list if (self.grid_dim in list or 0 in list) else None for list in all_states]
        terminating_states = [element for element in terminating_states if element is not None]
        terminating_states = torch.stack(terminating_states)
        n_terminating_states = terminating_states.size(0)
        return n_terminating_states
    
    @property
    def terminating_states(self) -> States:
        digits = torch.arange(self.grid_len)
        all_states = torch.cartesian_prod(*[digits] * self.grid_dim)
        terminating_states = [list if (self.grid_dim in list or 0 in list) else None for list in all_states]
        terminating_states = [element for element in terminating_states if element is not None]
        terminating_states = torch.stack(terminating_states)
        return self.States(terminating_states) 
