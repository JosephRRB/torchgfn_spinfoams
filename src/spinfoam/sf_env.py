import os
from typing import ClassVar, Tuple, cast

import torch
from torchtyping import TensorType

from gfn.envs.env import Env
from gymnasium.spaces import Discrete
from gfn.envs.preprocessors import KHotPreprocessor
from gfn.containers.states import States

from src.spinfoam.spinfoams import BaseSpinFoam

ROOT_DIR = os.path.abspath(__file__ + "/../../../")

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]


class SpinFoamEnvironment(Env):
    def __init__(
            self,
            spinfoam_model: BaseSpinFoam
    ):
        self.spinfoam_model = spinfoam_model
        self.device = self.spinfoam_model.device
        self.grid_len = int(2 * self.spinfoam_model.spin_j + 1)
        self.grid_dim = self.spinfoam_model.n_boundary_intertwiners
        self.log_reward_shift = self.spinfoam_model.estimate_log_sq_ampl_shift(frac=0.99)

        s0 = torch.zeros(
            self.grid_dim, dtype=torch.long, device=self.device
        )
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

        class SpinFoamGridStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.grid_dim,)
            s0 = env.s0
            sf = env.sf

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

        return SpinFoamGridStates

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        return actions == self.action_space.n - 1

    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), 1, reduce="add")

    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), -1, reduce="add")

    def log_reward(self, final_states: States) -> TensorFloat:
        final_states_raw = final_states.states_tensor
        log_sq_ampl = self.spinfoam_model.calculate_log_square_amplitudes(final_states_raw)
        log_rewards = log_sq_ampl - self.log_reward_shift
        return log_rewards.to(torch.float)