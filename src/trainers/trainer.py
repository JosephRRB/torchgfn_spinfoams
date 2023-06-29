import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

from gfn import LogitPBEstimator, LogitPFEstimator, LogZEstimator, LogStateFlowEstimator
from gfn.losses import TBParametrization, TrajectoryBalance, SubTBParametrization, SubTrajectoryBalance
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler, BackwardDiscreteActionsSampler

from src.losses.weighted_tb import WeightedTrajectoryBalance
from src.spinfoam.spinfoams import SingleVertexSpinFoam, StarModelSpinFoam
from src.spinfoam.sf_env import SpinFoamEnvironment


def train_gfn(
    spin_j,
    sf_model,
    generated_data_dir,
    hidden_dim=256,
    n_hidden_layers=2,
    activation_fn="relu",
    exploration_rate=0.5,
    learning_rate=0.0005,
    batch_size=int(1e3),
    n_iterations=int(1e4),
    evaluation_batch_size=int(1e6),
    generate_samples_every_m_training_samples=int(1e6),
):

    if sf_model == "single_vertex_model":
        spinfoam_model = SingleVertexSpinFoam(spin_j=spin_j)
    elif sf_model == "star_model":
        spinfoam_model = StarModelSpinFoam(spin_j=spin_j)
    else:
        raise ValueError(
            "Spinfoam model not yet implemented. "
            "Custom Spinfoam class can be made."
        )

    env = SpinFoamEnvironment(spinfoam_model=spinfoam_model)

    logit_PF = LogitPFEstimator(
        env=env,
        module_name="NeuralNet",
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        activation_fn=activation_fn
    )
    logit_PB = LogitPBEstimator(
        env=env,
        module_name="NeuralNet",
        torso=logit_PF.module.torso,  # To share parameters between PF and PB
    )
    logZ = LogZEstimator(torch.tensor(0.0))

    training_sampler = TrajectoriesSampler(
        env=env,
        actions_sampler=DiscreteActionsSampler(
            estimator=logit_PF,
            epsilon=exploration_rate
        )
    )
    eval_sampler = TrajectoriesSampler(
        env=env, actions_sampler=DiscreteActionsSampler(estimator=logit_PF)
    )

    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    loss_fn = TrajectoryBalance(
        parametrization=parametrization,
        log_reward_clip_min=-500.0
    )

    params = [
        {
            "params": [
                # val for key, val in parametrization.parameters.items() if "logZ" not in key
                val for key, val in parametrization.parameters.items() if ("PF" in key) or ("PB_last" in key)
            ],
            "lr": learning_rate,
        },
        {"params": [val for key, val in parametrization.parameters.items() if "logZ" in key], "lr": 0.1},
    ]
    optimizer = torch.optim.Adam(params=params)

    losses = []
    os.makedirs(generated_data_dir, exist_ok=True)

    for i in (pbar := tqdm(range(n_iterations))):
        trajectories = training_sampler.sample(
            n_trajectories=batch_size
        )

        optimizer.zero_grad()
        loss = loss_fn(trajectories)
        loss.backward()
        optimizer.step()

        trained_on_k_samples = (i + 1) * batch_size
        if trained_on_k_samples % generate_samples_every_m_training_samples == 0:
            pbar.set_postfix({"loss": loss.item()})
            eval_trajectories = eval_sampler.sample(
                n_trajectories=evaluation_batch_size
            )
            samples_file = Path(
                f"{generated_data_dir}/"
                f"epoch_{i + 1}"
                f"_after_learn_from_{trained_on_k_samples}"
                "_train_samples.csv"
            )
            samples_file.touch()

            np.savetxt(
                samples_file,
                eval_trajectories.last_states.states_tensor.numpy(),
                delimiter=",",
                fmt="%i",
            )
        losses.append(loss.item())
    return losses


def base_train_gfn(
    env,
    generated_data_dir,
    batch_size,
    n_iterations,
    hidden_dim=256,
    n_hidden_layers=2,
    activation_fn="relu",
    exploration_rate=0.0,
    learning_rate=0.0005,
):
    logit_PF = LogitPFEstimator(
        env=env,
        module_name="NeuralNet",
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        activation_fn=activation_fn
    )
    logit_PB = LogitPBEstimator(
        env=env,
        module_name="NeuralNet",
        torso=logit_PF.module.torso,
    )
    logZ = LogZEstimator(torch.tensor(0.0))

    training_sampler = TrajectoriesSampler(
        env=env,
        actions_sampler=DiscreteActionsSampler(
            estimator=logit_PF,
            epsilon=exploration_rate
        )
    )
    eval_sampler = TrajectoriesSampler(
        env=env, actions_sampler=DiscreteActionsSampler(estimator=logit_PF)
    )

    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    # loss_fn = TrajectoryBalance(
    #     parametrization=parametrization,
    #     log_reward_clip_min=-500.0
    # )
    loss_fn = WeightedTrajectoryBalance(
        parametrization=parametrization,
        log_reward_clip_min=-500.0
    )

    params = [
        {
            "params": [
                val for key, val in parametrization.parameters.items() if ("PF" in key) or ("PB_last" in key)
            ],
            "lr": learning_rate,
        },
        {"params": [val for key, val in parametrization.parameters.items() if "logZ" in key], "lr": 0.1},
    ]
    optimizer = torch.optim.Adam(params=params)

    losses = []
    os.makedirs(generated_data_dir, exist_ok=True)

    terminal_states = []
    for _ in tqdm(range(n_iterations)):
        trajectories = training_sampler.sample(
            n_trajectories=batch_size
        )

        optimizer.zero_grad()
        loss = loss_fn(trajectories)
        loss.backward()
        optimizer.step()

        if exploration_rate:
            eval_trajectories = eval_sampler.sample(
                n_trajectories=batch_size
            )
            states = eval_trajectories.last_states.states_tensor.numpy()
        else:
            states = trajectories.last_states.states_tensor.numpy()

        terminal_states.append(states)

        losses.append(loss.item())

    terminal_states = np.stack(terminal_states)

    np.save(
        f"{generated_data_dir}/terminal_states.npy",
        terminal_states
    )
    return terminal_states, losses

from gfn.containers.trajectories import Trajectories
from src.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer


LOSS_PARAMS = {
    "weighing": "geometric_within",
    "lambda": 0.9
}
REPLAY_PARAMS = {
    "capacity": 1000,
    "fraction_of_samples_from_top": 0.5,
    "top_and_bottom_fraction": 0.1
}
NN_PARAMS = {
    "hidden_dim": 256,
    "n_hidden_layers": 2,
    "activation_fn": "relu",
}


def train_gfn_subtb_with_replay(
    env,
    generated_data_dir,
    batch_size,
    n_iterations,
    learning_rate=0.0005,
    exploration_rate=0.0,
    forward_looking=True,
    loss_params=LOSS_PARAMS,
    replay_params=REPLAY_PARAMS,
    nn_params=NN_PARAMS,
):
    if replay_params:
        replay_buffer = PrioritizedReplayBuffer(
            env=env, **replay_params
        )
    else:
        replay_buffer = None

    logit_PF = LogitPFEstimator(
        env=env,
        module_name="NeuralNet",
        **nn_params
    )
    logit_PB = LogitPBEstimator(
        env=env,
        module_name="NeuralNet",
        torso=logit_PF.module.torso,
    )
    logF = LogStateFlowEstimator(
        env=env,
        module_name="NeuralNet",
        torso=logit_PF.module.torso,
        forward_looking=forward_looking
    )

    forward_sampler = TrajectoriesSampler(
        env=env,
        actions_sampler=DiscreteActionsSampler(
            estimator=logit_PF,
            epsilon=exploration_rate
        )
    )
    backward_sampler = TrajectoriesSampler(
        env=env, actions_sampler=BackwardDiscreteActionsSampler(estimator=logit_PB,)
    )
    eval_sampler = TrajectoriesSampler(
        env=env, actions_sampler=DiscreteActionsSampler(estimator=logit_PF)
    )

    parametrization = SubTBParametrization(logit_PF, logit_PB, logF)
    loss_fn = SubTrajectoryBalance(
        parametrization=parametrization,
        log_reward_clip_min=-500.0,
        **loss_params
    )

    params = [
        {
            "params": [
                val for key, val in parametrization.parameters.items() if ("PF_torso" in key) or ("last_layer" in key)
            ],
            "lr": learning_rate,
        },
    ]
    optimizer = torch.optim.Adam(params=params)

    losses = []
    os.makedirs(generated_data_dir, exist_ok=True)

    terminal_states = []
    for _ in tqdm(range(n_iterations)):
        if replay_buffer:
            replay_samples = replay_buffer.sample(batch_size)
            backward_trajectories = backward_sampler.sample_trajectories(replay_samples)
            offline_trajectories = Trajectories.revert_backward_trajectories(backward_trajectories)
            # padding with sf
            offline_trajectories.states.extend_with_sf(offline_trajectories.states.batch_shape[0] + 1)
        else:
            offline_trajectories = Trajectories(env=env)

        trajectories = forward_sampler.sample(n_trajectories=batch_size)
        if replay_params:
            replay_buffer.add(trajectories)
            trajectories.extend(offline_trajectories)

        optimizer.zero_grad()
        loss = loss_fn(trajectories)
        loss.backward()
        optimizer.step()

        if exploration_rate or replay_params:
            eval_trajectories = eval_sampler.sample(
                n_trajectories=batch_size
            )
            states = eval_trajectories.last_states.states_tensor.numpy()
        else:
            states = trajectories.last_states.states_tensor.numpy()

        terminal_states.append(states)

        losses.append(loss.item())

    terminal_states = np.stack(terminal_states)

    np.save(
        f"{generated_data_dir}/terminal_states.npy",
        terminal_states
    )
    return terminal_states, losses