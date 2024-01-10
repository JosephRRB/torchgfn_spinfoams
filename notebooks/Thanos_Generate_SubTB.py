import numpy as np
import os
import sys
from multiprocessing import Process, Pool
import time
import argparse


ROOT_DIR = os.path.abspath("__file__" + "/../../")
sys.path.insert(0, f"{ROOT_DIR}")

spin_j = 6

env_name = f"single vertex spinfoam/j={float(spin_j)}"
batch_size = 16
n_iterations = int(1e5)

vertex = np.load(f"{ROOT_DIR}/data/EPRL_vertices/Python/Dl_20/vertex_j_{float(spin_j)}.npz")
sq_ampl = vertex**2
grid_rewards = sq_ampl / np.sum(sq_ampl)


from src.MCMC.batched_mcmc import MCMCRunner

mcmc = MCMCRunner(grid_rewards=grid_rewards)
mcmc_chains, _ = mcmc.run_mcmc_chains(
    batch_size=batch_size, n_iterations=n_iterations, generated_data_dir=f"{ROOT_DIR}/thanos_data/MCMC/{env_name}"
)


from src.grid_environments.base import BaseGrid
from src.trainers.trainer import train_gfn
from itertools import product

loss_params = {
    "weighing": "DB",
    "lamda": 0.9,
}
replay_params = {
    "capacity": 1000,
    "fraction_of_samples_from_top": 0.5,
    "top_and_bottom_fraction": 0.1,
    "max_fraction_offline": 0.5,
}
nn_params = {
    "hidden_dim": 256,
    "n_hidden_layers": 2,
    "activation_fn": "relu",
}

device_str = "cpu" # cpu or cuda (torchgfn has some issues with gpu)

#if os.path.isfile("times"):
#    os.remove("times")


parametrization_name = "SubTB"
weighings = ["DB", "ModifiedDB", "equal", "geometric_within", "equal_within"] # "TB" and "geometric" do not work

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--position", help="use job number from slurm in python script for getting elements out of lists", type=int)

args = parser.parse_args()

weighing = weighings[args.position]

for exploration_rate in [0.01, 0.1, 0.9]:
    if weighing == "geometric_within" or weighing == "geometric":
            for lamda in [0.1, 0.9, 3]:
                start_time = time.time()
                loss_params = {
                    "weighing": weighing,
                    "lamda": lamda,
                }

            terminal_states, _ = train_gfn(
            env=BaseGrid(grid_rewards=grid_rewards, device_str=device_str),
            generated_data_dir=f"{ROOT_DIR}/thanos_data/GFN/{env_name}/{parametrization_name}", # need better way to label folders
            batch_size=batch_size,
            n_iterations=n_iterations,
            learning_rate=0.001,
            exploration_rate=exploration_rate, # (faster if set to 0)
            policy="sa",
            forward_looking=False,
            loss_params=loss_params,
            replay_params=None, # replay_params or None (faster if None)
            nn_params=nn_params,
            parametrization_name = parametrization_name
            )
    else:
        start_time = time.time()
        loss_params = {
                "weighing": weighing,
            }
                    
        terminal_states, _ = train_gfn(
        env=BaseGrid(grid_rewards=grid_rewards, device_str=device_str),
        generated_data_dir=f"{ROOT_DIR}/thanos_data/GFN/{env_name}/{parametrization_name}", # need better way to label folders
        batch_size=batch_size,
        n_iterations=n_iterations,
        learning_rate=0.001,
        exploration_rate=exploration_rate, # (faster if set to 0)
        policy="sa",
        forward_looking=False,
        loss_params=loss_params,
        replay_params=None, # replay_params or None (faster if None)
        nn_params=nn_params,
        parametrization_name = parametrization_name
    
    )           


    f = open("times", "a")
    if 'lamda' not in loss_params:
        f.write(f"\n{parametrization_name}, exploration rate {exploration_rate}, weighing {weighing} = {time.time()-start_time}")
    else:
        f.write(f"\n{parametrization_name}, exploration rate {exploration_rate}, weighing {weighing}, lamda {loss_params['lamda']} = {time.time()-start_time}")
    f.close()

    
'''
if __name__ == '__main__':
    processes = [Process(target=train_gfn, args=(
        BaseGrid(grid_rewards=grid_rewards, device_str=device_str),
        f"{ROOT_DIR}/thanos_data/GFN/{env_name}/{parametrization_name}", # need better way to label folders
        batch_size,
        n_iterations,
        0.001,
        0.0, # (faster if set to 0)
        "sa",
        True,
        loss_params,
        None, # replay_params or None (faster if None)
        nn_params,
        parametrization_name,)) for parametrization_name in ["DB", "TB", "ZVar", "SubTB", "FM"]]
    
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()


if __name__ == '__main__':
    pool = Pool()
    args = [(
        BaseGrid(grid_rewards=grid_rewards, device_str=device_str),
        f"{ROOT_DIR}/thanos_data/GFN/{env_name}/{parametrization_name}", # need better way to label folders
        batch_size,
        n_iterations,
        0.001,
        0.0, # (faster if set to 0)
        "sa",
        True,
        loss_params,
        None, # replay_params or None (faster if None)
        nn_params,
        parametrization_name) for parametrization_name in ["DB", "TB", "ZVar", "SubTB", "FM"]]
    
    pool.starmap(train_gfn, args)
                  
    pool.close()
    pool.join()
    

def train(parametrization_name):
    return train_gfn(env=BaseGrid(grid_rewards=grid_rewards, device_str=device_str),
        generated_data_dir=f"{ROOT_DIR}/thanos_data/GFN/{env_name}/{parametrization_name}", # need better way to label folders
        batch_size=batch_size,
        n_iterations=n_iterations,
        learning_rate=0.001,
        exploration_rate=exploration_rate, # (faster if set to 0)
        policy="sa",
        forward_looking=False,
        loss_params=loss_params,
        replay_params=None, # replay_params or None (faster if None)
        nn_params=nn_params,
        parametrization_name = parametrization_name
             )

if __name__ == '__main__':
    pool = Pool(5)
    
    print("Created pool!")
    
    async_results = [pool.apply_async(train, args=(parametrization_name,)) for parametrization_name in ["DB", "TB", "ZVar", "SubTB", "FM"]]
    
    print("Created processes.")
    
    results = [ar.get() for ar in async_results] 
    
    pool.close()
    pool.join()



def train(parametrization_name):
    return train_gfn(env=BaseGrid(grid_rewards=grid_rewards, device_str=device_str),
        generated_data_dir=f"{ROOT_DIR}/thanos_data/GFN/{env_name}/{parametrization_name}", # need better way to label folders
        batch_size=batch_size,
        n_iterations=n_iterations,
        learning_rate=0.001,
        exploration_rate=exploration_rate, # (faster if set to 0)
        policy="sa",
        forward_looking=False,
        loss_params=loss_params,
        replay_params=None, # replay_params or None (faster if None)
        nn_params=nn_params,
        parametrization_name = parametrization_name
                    )
                     
if __name__ == '__main__':
    with Pool(5) as pool:
        results = pool.map(train, ["DB", "TB", "ZVar", "SubTB", "FM"])
        
'''