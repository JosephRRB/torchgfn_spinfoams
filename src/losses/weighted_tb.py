import torch
from torchtyping import TensorType

from gfn.containers import Trajectories
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.losses import TBParametrization
from gfn.samplers.actions_samplers import (
    BackwardDiscreteActionsSampler,
    DiscreteActionsSampler,
)

LossTensor = TensorType[0, float]


class WeightedTrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: TBParametrization,
        log_reward_clip_min: float = -12,
        on_policy: bool = False,
    ):
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )
        self.on_policy = on_policy

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        _, _, scores = self.get_trajectories_scores(trajectories)
        log_rewards = trajectories.log_rewards.clamp_min(self.log_reward_clip_min)
        weights = torch.softmax(log_rewards, dim=-1)
        loss_per_trajectory = (scores + self.parametrization.logZ.tensor).pow(2)
        weighted_loss = weights*loss_per_trajectory
        loss = weighted_loss.sum()
        if torch.isnan(loss):
            raise ValueError("loss is nan")
        return loss