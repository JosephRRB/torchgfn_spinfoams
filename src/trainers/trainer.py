import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

from gfn import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.losses import TBParametrization, TrajectoryBalance
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler

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

    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    loss_fn = TrajectoryBalance(
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

        terminal_states.append(
            trajectories.last_states.states_tensor.numpy()
        )

        losses.append(loss.item())

    terminal_states = np.stack(terminal_states)

    np.save(
        f"{generated_data_dir}/terminal_states.npy",
        terminal_states
    )
    return terminal_states, losses