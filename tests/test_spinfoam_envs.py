import numpy as np
import torch

from src.spinfoam.spinfoams import SingleVertexSpinFoam, StarModelSpinFoam
from src.spinfoam.sf_env import SpinFoamEnvironment


def test_stored_single_vertex_amplitudes_are_correct():
    spin_j = 3.0
    env = SpinFoamEnvironment(
        spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j)
    )
    positions = torch.tensor([[0, 3, 0, 2, 0]])
    vertex_amplitude = env.spinfoam_model.get_spinfoam_amplitudes(positions)

    # From Python_notebook.ipynb
    expected_amplitude = -5.071973704515683e-13

    assert vertex_amplitude == expected_amplitude


def test_calculated_single_vertex_log_square_amplitudes_are_correct():
    spin_j = 3.0

    env = SpinFoamEnvironment(
        spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j)
    )
    positions = torch.tensor([[0, 3, 0, 2, 0]])
    log_sq_ampl = env.spinfoam_model.calculate_log_square_amplitudes(positions)

    # From Python_notebook.ipynb
    expected_amplitude = torch.tensor([-5.071973704515683e-13])
    expected_log_sq_ampl = 2*torch.log(torch.abs(expected_amplitude))

    assert log_sq_ampl.to(torch.float) == expected_log_sq_ampl


def test_calculated_single_vertex_log_rewards_are_correct():
    spin_j = 3.0

    env = SpinFoamEnvironment(
        spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j)
    )
    SF_States = env.make_States_class()
    states = SF_States(torch.tensor([[0, 3, 0, 2, 0]]))

    # Define log rewards as
    log_rewards = env.log_reward(states)

    # From Python_notebook.ipynb
    expected_amplitude = torch.tensor([-5.071973704515683e-13])
    log_sq_ampl = 2*torch.log(torch.abs(expected_amplitude))
    # Heuristic constant shift for log rewards
    log_sq_ampl_max = 0.99*2*env.spinfoam_model.n_vertices*torch.log(torch.max(
        torch.abs(env.spinfoam_model.single_vertex_amplitudes)
    ))
    expected_log_rewards = log_sq_ampl - log_sq_ampl_max
    torch.testing.assert_allclose(log_rewards, expected_log_rewards)


def test_calculated_star_model_log_square_amplitudes_are_correct():
    spin_j = 3.0
    env = SpinFoamEnvironment(
        spinfoam_model=StarModelSpinFoam(spin_j=spin_j)
    )
    positions = torch.tensor([
        [0, 1, 4, 3, 4, 0, 1, 4, 3, 2, 0, 1, 4, 3, 0, 0, 1, 4, 3, 0],
        [2, 0, 1, 4, 3, 0, 0, 1, 4, 3, 0, 0, 1, 4, 3, 4, 0, 1, 4, 3]
    ])

    vertex_amplitudes = env.spinfoam_model.single_vertex_amplitudes
    log_sq_ampls = env.spinfoam_model.calculate_log_square_amplitudes(positions)

    # From Python_notebook.ipynb
    def star_reward_optimized(tensor, indices, optimize_path=False):
        return np.square(
            np.einsum(
                'abcde, e, d, c, b, a ->', tensor,
                tensor[indices[0], indices[1], indices[2], indices[3], :],
                tensor[indices[4], indices[5], indices[6], indices[7], :],
                tensor[indices[8], indices[9], indices[10], indices[11], :],
                tensor[indices[12], indices[13], indices[14], indices[15], :],
                tensor[indices[16], indices[17], indices[18], indices[19], :],
                optimize=optimize_path
            )
        )

    expected_sq_star_amplitude_0 = star_reward_optimized(
        vertex_amplitudes.numpy(), positions[0, :].numpy()
    )
    expected_sq_star_amplitude_1 = star_reward_optimized(
        vertex_amplitudes.numpy(), positions[1, :].numpy()
    )

    expected_log_sq_ampl_0 = np.log(expected_sq_star_amplitude_0)
    expected_log_sq_ampl_1 = np.log(expected_sq_star_amplitude_1)

    np.testing.assert_almost_equal(
        log_sq_ampls[0], expected_log_sq_ampl_0
    )
    np.testing.assert_almost_equal(
        log_sq_ampls[1], expected_log_sq_ampl_1
    )


def test_calculated_star_model_rewards_are_correct():
    spin_j = 3.0
    env = SpinFoamEnvironment(
        spinfoam_model=StarModelSpinFoam(spin_j=spin_j)
    )

    positions = torch.tensor([
        [0, 1, 4, 3, 4, 0, 1, 4, 3, 2, 0, 1, 4, 3, 0, 0, 1, 4, 3, 0],
        [2, 0, 1, 4, 3, 0, 0, 1, 4, 3, 0, 0, 1, 4, 3, 4, 0, 1, 4, 3]
    ])
    SF_States = env.make_States_class()
    states = SF_States(positions)
    log_rewards = env.log_reward(states)

    vertex_amplitudes = env.spinfoam_model.single_vertex_amplitudes
    # Heuristic constant shift for log rewards
    log_sq_ampl_max = 0.99 * 2 * env.spinfoam_model.n_vertices * torch.log(
        torch.max(torch.abs(vertex_amplitudes))
    ).numpy()

    # From Python_notebook.ipynb
    def star_reward_optimized(tensor, indices, optimize_path=False):
        return np.square(
            np.einsum(
                'abcde, e, d, c, b, a ->', tensor,
                tensor[indices[0], indices[1], indices[2], indices[3], :],
                tensor[indices[4], indices[5], indices[6], indices[7], :],
                tensor[indices[8], indices[9], indices[10], indices[11], :],
                tensor[indices[12], indices[13], indices[14], indices[15], :],
                tensor[indices[16], indices[17], indices[18], indices[19], :],
                optimize=optimize_path
            )
        )

    expected_sq_star_amplitude_0 = star_reward_optimized(
        vertex_amplitudes.numpy(), positions[0, :].numpy()
    )
    expected_sq_star_amplitude_1 = star_reward_optimized(
        vertex_amplitudes.numpy(), positions[1, :].numpy()
    )

    expected_log_sq_ampl_0 = np.log(expected_sq_star_amplitude_0)
    expected_log_sq_ampl_1 = np.log(expected_sq_star_amplitude_1)

    expected_reward_0 = expected_log_sq_ampl_0 - log_sq_ampl_max
    expected_reward_1 = expected_log_sq_ampl_1 - log_sq_ampl_max

    np.testing.assert_almost_equal(
        log_rewards[0], expected_reward_0
    )
    np.testing.assert_almost_equal(
        log_rewards[1], expected_reward_1
    )

