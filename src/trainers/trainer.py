import os
from typing import Literal
from tqdm import tqdm

import numpy as np
import torch

from gfn import LogitPBEstimator, LogitPFEstimator, LogStateFlowEstimator
from gfn.losses import SubTBParametrization, SubTrajectoryBalance

from gfn.losses.detailed_balance import DBParametrization, DetailedBalance
from gfn.losses.flow_matching import FlowMatching, FMParametrization
from gfn.losses.sub_trajectory_balance import SubTBParametrization, SubTrajectoryBalance
from gfn.losses.trajectory_balance import (
    LogPartitionVarianceLoss,
    PFBasedParametrization,
    TBParametrization,
    TrajectoryBalance,
)
    
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler, BackwardDiscreteActionsSampler
from gfn.containers.trajectories import Trajectories

from src.policies.ssr_policy import ForwardLogRelativeEdgeFlowEstimator, BackwardLogRelativeEdgeFlowEstimator
from src.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer


LOSS_PARAMS = {
    "weighing": "geometric_within",
    "lamda": 0.9,
}
REPLAY_PARAMS = {
    "capacity": 1000,
    "fraction_of_samples_from_top": 0.5,
    "top_and_bottom_fraction": 0.1,
    "max_fraction_offline": 0.5,
}
NN_PARAMS = {
    "hidden_dim": 256,
    "n_hidden_layers": 2,
    "activation_fn": "relu",
}


def train_gfn(
    env,
    generated_data_dir,
    batch_size,
    n_iterations,
    learning_rate=0.0005,
    exploration_rate=0.0,
    policy: Literal["sa", "ssr"] = "sa",
    forward_looking=False,
    loss_params=LOSS_PARAMS,
    replay_params=REPLAY_PARAMS, # replay_params or None
    nn_params=NN_PARAMS,
    parametrization_name = "SubTB"
):
    if replay_params:
        max_fraction_offline = replay_params.pop("max_fraction_offline")
        replay_buffer = PrioritizedReplayBuffer(
            env=env, **replay_params
        )
    else:
        replay_buffer = None

    if policy == "sa":
        forward_policy = LogitPFEstimator(
            env=env,
            module_name="NeuralNet",
            **nn_params
        )
        backward_policy = LogitPBEstimator(
            env=env,
            module_name="NeuralNet",
            torso=forward_policy.module.torso,
        )
        logF = LogStateFlowEstimator(
            env=env,
            module_name="NeuralNet",
            torso=forward_policy.module.torso,
            forward_looking=forward_looking
        )
    elif policy == "ssr":
        forward_policy = ForwardLogRelativeEdgeFlowEstimator(
            env=env, **nn_params
        )
        backward_policy = BackwardLogRelativeEdgeFlowEstimator(
            env=env, **nn_params
        )
        logF = LogStateFlowEstimator(
            env=env,
            module_name="NeuralNet",
            forward_looking=forward_looking
        )
    else:
        raise NotImplementedError("Only 'sa' and 'ssr' policies are available")

    forward_sampler = TrajectoriesSampler(
        env=env,
        actions_sampler=DiscreteActionsSampler(
            estimator=forward_policy,
            epsilon=exploration_rate
        )
    )
    backward_sampler = TrajectoriesSampler(
        env=env, actions_sampler=BackwardDiscreteActionsSampler(estimator=backward_policy,)
    )
    eval_sampler = TrajectoriesSampler(
        env=env, actions_sampler=DiscreteActionsSampler(estimator=forward_policy)
    )
    
    if parametrization_name == "DB":
        parametrization = DBParametrization(forward_policy, backward_policy, logF)
        loss_cls = DetailedBalance
    elif parametrization_name == "TB":
        parametrization = TBParametrization(forward_policy, backward_policy, logZ)
        loss_cls = TrajectoryBalance
    elif parametrization_name == "ZVar":
        parametrization = PFBasedParametrization(forward_policy, backward_policy)
        loss_cls = LogPartitionVarianceLoss
    elif parametrization_name == "SubTB":
        parametrization = SubTBParametrization(forward_policy, backward_policy, logF)
        loss_cls = SubTrajectoryBalance
    elif parametrization_name == "FM":
        parametrization = parametrization = FMParametrization(forward_policy)
        loss_cls = FlowMatching
    else:
        raise ValueError(f"Unknown parametrization {parametrization_name}")


    if policy == "sa":
        params = [
            val for key, val in parametrization.parameters.items()
            if ("PF_torso" in key) or ("last_layer" in key)
        ]
    else:
        params = list(parametrization.parameters.values())

    optimizer = torch.optim.Adam(
        params=[{"params": params, "lr": learning_rate,}]
    )

    losses = []
    terminal_states = []
    for _ in tqdm(range(n_iterations)):
        if replay_buffer:
            max_n_samples_offline = int(max_fraction_offline * batch_size)
            replay_samples = replay_buffer.sample(max_n_samples_offline)
            backward_trajectories = backward_sampler.sample_trajectories(replay_samples)
            offline_trajectories = Trajectories.revert_backward_trajectories(backward_trajectories)
            # padding with sf
            offline_trajectories.states.extend_with_sf(offline_trajectories.states.batch_shape[0] + 1)
        else:
            offline_trajectories = Trajectories(env=env)
            offline_trajectories.actions = offline_trajectories.actions.to(env.device)
            offline_trajectories.log_probs = offline_trajectories.log_probs.to(env.device)
            offline_trajectories.when_is_done = offline_trajectories.when_is_done.to(env.device)


        n_samples_online = batch_size - offline_trajectories.n_trajectories
        trajectories = forward_sampler.sample(n_trajectories=n_samples_online)
        
        if parametrization_name == "DB":
            training_objects = trajectories.to_transitions()
        elif parametrization_name == "FM":
            training_objects = trajectories.to_non_initial_intermediary_and_terminating_states()
        else:
            training_objects = trajectories
        
        if replay_params:
            replay_buffer.add(trajectories)
            # Pad log_probs to avoid error when combining with forward trajectories
            # log_probs should not be used in loss
            offline_trajectories.log_probs = torch.cat([
                offline_trajectories.log_probs,
                torch.full(
                    size=(
                        1,
                        offline_trajectories.n_trajectories,
                    ),
                    fill_value=0,
                    dtype=torch.float,
                ).to(env.device)
            ], dim=0)
            trajectories.extend(offline_trajectories)

        optimizer.zero_grad()
        
        if parametrization_name == "FM":
            loss_fn = loss_cls(parametrization=parametrization)
        else:
            loss_fn = loss_cls(parametrization=parametrization, **loss_params)
        
        loss = loss_fn(training_objects)
        
        if parametrization_name == "TB":
            assert torch.all(
                torch.abs(
                    loss_fn.get_pfs_and_pbs(training_objects)[0]
                    - training_objects.log_probs
                )
                < 1e-5
            )
        
        loss.backward()
        
        
        optimizer.step()

        if exploration_rate or replay_params:
            eval_trajectories = eval_sampler.sample(
                n_trajectories=batch_size
            )
            states = eval_trajectories.last_states.states_tensor.cpu().numpy()
        else:
            states = trajectories.last_states.states_tensor.cpu().numpy()

        terminal_states.append(states)

        losses.append(loss.item())

    terminal_states = np.stack(terminal_states)

    os.makedirs(generated_data_dir, exist_ok=True)
    np.save(
        f"{generated_data_dir}/terminal_states.npy",
        terminal_states
    )
    if replay_buffer:
        np.save(
            f"{generated_data_dir}/replay_states.npy",
            replay_buffer.terminating_states.states_tensor.cpu().numpy()
        )
        np.save(
            f"{generated_data_dir}/replay_log_rewards.npy",
            replay_buffer.terminating_states.log_rewards.cpu().numpy()
        )

    return terminal_states, losses