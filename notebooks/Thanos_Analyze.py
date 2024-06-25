# Analyze and save the results
import numpy as np
import os
import sys
from tqdm import tqdm


ROOT_DIR = os.path.abspath("__file__" + "/../../")
sys.path.insert(0, f"{ROOT_DIR}")

spin_j = 6

env_name = f"single vertex spinfoam/j={float(spin_j)}"
batch_size = 16
n_iterations = int(1e5)

vertex = np.load(f"{ROOT_DIR}/data/EPRL_vertices/Python/Dl_20/vertex_j_{float(spin_j)}.npz")
sq_ampl = vertex**2
grid_rewards = sq_ampl / np.sum(sq_ampl)

import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_observable_expectation_values(
    empirical_distributions, n_samples_used, spin_j, ax, label
):
    intertwiners = np.arange(int(2*spin_j + 1))
    cos_angle = intertwiners*(intertwiners+1)/(2*spin_j*(spin_j+1)) - 1

    reduced_distr_over_t = np.sum(empirical_distributions, axis=(2, 3, 4, 5))
    mean_cos_angles = np.sum(reduced_distr_over_t*cos_angle, axis=1)

    ax.scatter(n_samples_used, mean_cos_angles, label=label)
    ax.hlines(-0.33333, n_samples_used[0], n_samples_used[-1], colors='k', linestyles='dashed')
    ax.set_xlabel("Number of samples in distribution")
    ax.set_ylabel(r"<cos $\theta$>")
    #ax.set_ylim(-0.5, -0.1)
    ax.legend()
    
def get_distributions_over_time(grid_positions, grid_len, every_n_iterations=100):
    n_iterations, _, grid_dim = grid_positions.shape

    counts = np.zeros(shape=(grid_len, )*grid_dim)
    n_samples = 0
    empirical_distributions_over_time = []
    n_samples_used_over_time = []
    for i in range(0, n_iterations, every_n_iterations):
        states = np.concatenate(grid_positions[i:i+every_n_iterations])

        n_samples += states.shape[0]
        np.add.at(counts, tuple(states.T), 1)
        
        empirical_distributions_over_time.append(counts/n_samples)
        n_samples_used_over_time.append(n_samples)
    empirical_distributions_over_time = np.stack(empirical_distributions_over_time)
    return empirical_distributions_over_time, n_samples_used_over_time

def plot_l1_errors(
    empirical_distributions_over_time, expected_distribution, n_samples_used, ax, label
):
    l1_errors = np.abs(empirical_distributions_over_time - expected_distribution)
    
    grid_axes = empirical_distributions_over_time.ndim
    ave_error_over_time = np.mean(l1_errors, axis=tuple(range(1, grid_axes)))
    ax.scatter(n_samples_used, ave_error_over_time, label=label)
    ax.set_xlabel("Number of samples in distribution")
    ax.set_ylabel(r"Average Distribution Error")
    ax.legend()
    
def plot_log_empirical_vs_log_expected(empirical, expected, ax, label):
    log_empirical = np.clip(np.log(empirical.ravel()), a_min=-50, a_max=None)
    log_expected = np.log(expected.ravel())
    
    expected_range = [log_expected.min(), log_expected.max()]
    ax.scatter(log_empirical, log_expected, label=label)
    ax.plot(expected_range, expected_range, ls="--", color="r")
    ax.set_xlabel("log P")
    ax.set_ylabel(r"log $A^2$")
    ax.legend()
    
def plot_observable_expectation_values_window(
    empirical_distributions, iterations, spin_j, ax, label
):
    intertwiners = np.arange(int(2*spin_j + 1))
    cos_angle = intertwiners*(intertwiners+1)/(2*spin_j*(spin_j+1)) - 1

    reduced_distr_over_t = np.sum(empirical_distributions, axis=tuple(range(2, empirical_distributions.ndim)))
    mean_cos_angles = np.sum(reduced_distr_over_t*cos_angle, axis=1)

    ax.scatter(iterations, mean_cos_angles, label=label)
    ax.hlines(-0.33333, iterations[0], iterations[-1], colors='k', linestyles='dashed')
    ax.set_xlabel("iteration number")
    ax.set_ylabel(r"<cos $\theta$>")
    ax.set_ylim(-0.5, -0.1)
    ax.legend()
    
def plot_l1_errors_window(
    empirical_distributions_over_time, expected_distribution, iterations, ax, label
):
    l1_errors = np.abs(empirical_distributions_over_time - expected_distribution)
    
    grid_axes = empirical_distributions_over_time.ndim
    ave_error_over_time = np.mean(l1_errors, axis=tuple(range(1, grid_axes)))
    ax.scatter(iterations, ave_error_over_time, label=label)
    ax.set_xlabel("iteration number")
    ax.set_ylabel("Average Distribution Error")
    ax.legend()
    
def get_distributions_over_time_window(grid_positions, grid_len, window_size=100, every_n_iterations=100):
    n_iterations, _, grid_dim = grid_positions.shape

    # counts = np.zeros(shape=(grid_len, )*grid_dim)
    empirical_distributions_over_time = []
    iterations = []
    
    w_idx, w_mod = divmod(window_size, 2)
    
    for i in range(0, n_iterations, every_n_iterations):
        left_idx = max(0, i - w_idx)
        right_idx = min(n_iterations, i + w_idx + w_mod)
        if left_idx == 0:
            right_idx = window_size
        if right_idx == n_iterations:
            left_idx = n_iterations - window_size

        states = np.concatenate(grid_positions[left_idx:right_idx])
        num_samples = states.shape[0]
        
        counts = np.zeros(shape=(grid_len, )*grid_dim)
        np.add.at(counts, tuple(states.T), 1)
        empirical_distributions_over_time.append(counts/num_samples)
        iterations.append(i)

    empirical_distributions_over_time = np.stack(empirical_distributions_over_time)
    return empirical_distributions_over_time, iterations

def get_distributions_over_time_flattened(grid_len, distributions_over_time, iteration = -1):
    distributions_over_time_flattened = []
    grid_coordinates = []
    if distributions_over_time.ndim == 6:
        distributions_over_time = distributions_over_time[iteration, :, :, :, :, :]
    for i1 in range(grid_len):
        for i2 in range(grid_len):
            for i3 in range(grid_len):
                for i4 in range(grid_len):
                    for i5 in range(grid_len):
                        distributions_over_time_flattened.append(distributions_over_time[i1, i2, i3, i4, i5])
                       
                        grid_coordinates.append(np.array([i1, i2, i3, i4, i5]))

    return grid_coordinates, np.array(distributions_over_time_flattened)

#Return neural net parameters for the specific case
def case_parameters(str):
    lst = str.split("_")
    if lst[0] == "SubTB" and len(lst) == 3: #no lamda
        return f"Parametrization = {lst[0]}, Exploration Rate = {lst[1]}, Weighing = {lst[2]}"
    elif lst[0] == "SubTB" and len(lst) > 3: #with lamda
        return f"Parametrization = {lst[0]}, Exploration Rate = {lst[1]}, Weighing = {lst[2]}, λ = {lst[3]}"
    else: #not SubTB
        return f"Parametrization = {lst[0]}, Exploration Rate = {lst[1]}"

window_size = int(n_iterations / 10) if int(n_iterations / 10) > 0 else 1

grid_len = int(2*spin_j + 1)

every_n_iterations = int(n_iterations / 10**2) if int(n_iterations / 10**2) > 0 else 1


# Load MCMC chains
mcmc_chains = np.load(f"{ROOT_DIR}/thanos_data/MCMC/single vertex spinfoam/j={float(spin_j)}/mcmc_chains.npy")

mcmc_distributions_over_time, mcmc_n_t = get_distributions_over_time(
    mcmc_chains, grid_len, every_n_iterations=every_n_iterations, 
)

mcmc_distributions_over_time_window, mcmc_iterations = get_distributions_over_time_window(
    mcmc_chains, grid_len, window_size=window_size, every_n_iterations=every_n_iterations
)


_, mcmc_distributions_over_time_flattened = get_distributions_over_time_flattened(
    grid_len, mcmc_distributions_over_time
)


# Load GFN states
thedir=f"{ROOT_DIR}/thanos_data/GFN/single vertex spinfoam/j={float(spin_j)}/"

for parametrization in tqdm(next(os.walk(thedir))[1]):
    
    print(f"Parametrization is: {parametrization}")
    
    if parametrization == "Figures":
        continue
    
    #Parameters for the figures' title
    parameters = case_parameters(parametrization)
    
    print(f"{parameters}")
    
    gfn_states = np.load(thedir+parametrization+"/terminal_states.npy")
    
    gfn_distributions_over_time, gfn_n_t = get_distributions_over_time(
        gfn_states, grid_len, every_n_iterations=every_n_iterations
    )

    gfn_distributions_over_time_window, gfn_iterations = get_distributions_over_time_window(
        gfn_states, grid_len, window_size=window_size, every_n_iterations=every_n_iterations
    )
    
    print("\tFinished loading states!")

    # Plots with windows

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    fig.suptitle(f"j={spin_j}, " + parameters)
    
    plot_l1_errors_window(
        gfn_distributions_over_time_window, grid_rewards, gfn_iterations, ax[0], "GFN"
    )
    
    plot_l1_errors_window(
        mcmc_distributions_over_time_window, grid_rewards, mcmc_iterations, ax[0], "MCMC"
    )

    plot_observable_expectation_values_window(
        gfn_distributions_over_time_window, gfn_iterations, spin_j, ax[1], "GFN"
    )
    
    plot_observable_expectation_values_window(
        mcmc_distributions_over_time_window, mcmc_iterations, spin_j, ax[1], "MCMC"
    )

    plt.tight_layout()
    
    plt.savefig(thedir+parametrization+"/Loss_Cos_Window.png", bbox_inches='tight')
    
    plt.close()
    
    print("\tFinished plots with windows!")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    fig.suptitle(f"j={spin_j}, " + parameters)
    
    plot_l1_errors(
        gfn_distributions_over_time, grid_rewards, gfn_n_t, ax[0], "GFN"
    )
    plot_l1_errors(
        mcmc_distributions_over_time, grid_rewards, mcmc_n_t, ax[0], "MCMC"
    )

    plot_observable_expectation_values(
        gfn_distributions_over_time, gfn_n_t, spin_j, ax[1], "GFN"
    )
    plot_observable_expectation_values(
        mcmc_distributions_over_time, mcmc_n_t, spin_j, ax[1], "MCMC"
    )

    plt.tight_layout()

    plt.savefig(thedir+parametrization+"/Loss_Cos.png", bbox_inches='tight')
    
    plt.close()

    
    print("\tFinished plots! >.<")

    # Plot the states with respect to the Euclidean distance
    # Create a 2D array of the grid coordinates and the respective rewards, flattened.
    grid_coordinates_list, gfn_distributions_over_time_flattened = get_distributions_over_time_flattened(
        grid_len, gfn_distributions_over_time
    )

    grid_coordinates_list, mcmc_distributions_over_time_flattened = get_distributions_over_time_flattened(
        grid_len, mcmc_distributions_over_time
    )

    _, theoretical_distributions_over_time_flattened = get_distributions_over_time_flattened(
        grid_len, grid_rewards
    )

    df = pd.DataFrame(
        {
            "Coordinates": grid_coordinates_list,
            "Theoretical": theoretical_distributions_over_time_flattened,
            "MCMC": mcmc_distributions_over_time_flattened,
            "GFN": gfn_distributions_over_time_flattened
         }
    )

    df["distance"] = df["Coordinates"].apply(lambda x: np.sqrt(sum(x**2)))
    s = 3
    alpha = 0.5

    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(20, 12))
    
    fig.suptitle(f"j={spin_j}, " + parameters)    
    
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax4.set_yscale("log")

    sns.scatterplot(x=df["distance"], y=df[df.Theoretical > 10**(-6)].Theoretical, label="Theoretical", s=s, alpha=alpha, ax=ax1, color="blue")
    sns.scatterplot(x=df["distance"], y=df["MCMC"], label="MCMC", s=s, alpha=alpha, ax=ax2, color="red")
    sns.scatterplot(x=df["distance"], y=df["GFN"], label="GFN", s=s, alpha=alpha, ax=ax3, color="green")
    
    sns.scatterplot(x=df["distance"], y=df[df.Theoretical > 10**(-6)].Theoretical, label="Theoretical", s=s, alpha=alpha, ax=ax4, color="blue")
    sns.scatterplot(x=df["distance"], y=df["MCMC"], label="MCMC", s=s, alpha=alpha, ax=ax4, color="red")
    sns.scatterplot(x=df["distance"], y=df["GFN"], label="GFN", s=s, alpha=alpha, ax=ax4, color="green")
    
    plt.savefig(thedir+parametrization+"/Euclidean_Distance.png", bbox_inches='tight')
    
    plt.close()
    
    print("\tFinished Euclidean plots!")
    print(f"Done with {parametrization}.")