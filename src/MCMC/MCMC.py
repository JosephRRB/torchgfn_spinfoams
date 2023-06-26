import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import norm

def grid_rewards_2d(grid_length=13, r0=0.1, r1=0.5, r2=2.0):
    """
    Calculate a 2D grid reward fucntion.
    
    Parameters
    ----------

    grid_length: (float)
                - The length of the grid.
    
    r0: (float)
                - Height of the valleys searating the peaks.

    r1: (float)
                - Mid level height near the peaks has height r0 + r1.

    r2: (float)
                - Height of the peaks.
                 
    Return:
        -------
    rewards: (array)
                - A 2D array with the corresponding values of the function in the 2D grid of length grid_length.
                      
    """

    assert grid_length >= 7
    
    coord_1d_dist_from_center = np.abs(np.arange(grid_length) / (grid_length - 1) - 0.5)
    mid_level_1d = (coord_1d_dist_from_center > 0.25).astype(float)
    mid_level_2d = mid_level_1d[None, :]*mid_level_1d[:, None]
    high_level_1d = ((coord_1d_dist_from_center > 0.3) * (coord_1d_dist_from_center < 0.4)).astype(float)
    high_level_2d = high_level_1d[None, :]*high_level_1d[:, None]

    rewards = r0 + r1*mid_level_2d + r2*high_level_2d
    return  rewards


def discrete_normal_distribution(grid_length, mean=0.0, deviation=1.0):
    """
    Calculate the normal disctribution of discrete variables. 
    
    Parameters
    ----------

    grid_length: (integer)
                - Number of points drawn from the normal distribution.
                - Number of points returned.


    mean: (float)
                - The location of the normal distribution.

    deviation: (float)
                - The deviation of the noirmal distribution.

    Return:
        -------
        truncated_coefficients: (array)
                 - The probability density function of a truncated discrete normal distribution.
                 - Its number is given by the grid_length. 
                      
    """
    truncated_coefficients = np.zeros(grid_length, dtype=np.float64)

    for i in range(0,grid_length):
        cdf_differences = 0.0
        for n in range(-i, grid_length-i):
            cdf_differences += (norm.cdf(n + 0.5, loc=mean, scale=deviation) - norm.cdf(n - 0.5, loc=mean, scale=deviation))

        truncated_coefficients[i] = cdf_differences
        
    return truncated_coefficients/np.sum(truncated_coefficients)


def VertexMCMC(grid_length, iterations_number, batch_size, burn_factor, verbosity, draws_folder, deviation, mean = 0., reward_function = grid_rewards_2d, dimensions = 2):

    """
    Run the MCMC simulation and save results in the corresponding folder. 
    
    Parameters
    ----------
   
    iterations_number: (integer)
                - The number of iterations for the MCMC.

    batch_size: (integer)
                - The number of iterations per batch.

    burn_factor: (integer)
                - The number of iterations discarded in each batch due to non-convergence.

    verbosity: (integer)
                - Whether information should be printed.
    
    draws_folder: (string)
                - The lcoation of the data folder.
    
    deviation: (float)
                - The standard deviation of our Gaussian distribution.

    mean: (float)
                - The mean of our Gaussian distribution.

    reward_function: (function)
                - The reward function our MCMC is trying to lean from. 
                - Should return a 'dimensions' times array where is dimension has grid_length.

    dimensions: (integer)
                - The number of dimensions.
         
    """

    # Make sure number of iterations is an integer multiple of the batch size.
    if(iterations_number % batch_size != 0):
        raise ValueError("Number of iterations must be a multiple of batch size.")
    
    
    # The probability of each state happening in the gid. discrete_normal_distribution gives an almost unifrom sitribution for every grid point.
    truncated_coefficients = discrete_normal_distribution(grid_length=grid_length, mean=mean, deviation=deviation)

    draw = np.zeros(dimensions + 1, dtype=np.int64) # The current draw (state).
    gaussian_draw = np.zeros(dimensions , np.int64) # The guassian deformation to be added in the current draw (state).

    prob = 0.0 # Value of the reward function for the current draw.
  
    while( prob == 0 ):
        for i in range(np.size(draw)):
            draw[i] = np.random.randint(grid_length) # Initialize current draw.
        
        position_probability = reward_function(grid_length) 
        prob = position_probability[tuple(draw[:-1])]
        
    draw[-1] = 1 # Initial multiplicity

    if (verbosity > 1):
        print("Initial draw is", draw[:-1],"with prob", prob)

    # Proposed draw (to be determinned adding the guassian deformation to the current draw).
    proposed_draw = np.zeros(dimensions , dtype=np.int64)

    acceptance_ratio = 0 # To measure the percentage of accepted moves by the simulator. _estimator of the codes optimization.
    multiplicity = 1    # Initial multiplicity counter.
    
    RW_monitor = True

    df_all_batches = pd.DataFrame({"acceptance rate (%)":[], "run time (s)":[]}) # DataFrame to save universal data for the each batch.

    batch_counter = 0 # Indicator to be used naming each batch.

    draws = np.empty( shape=(0,dimensions + 1)) # Array to save the states visited in each batch.

    all_batches_statistics = np.zeros(shape=(0,2)) # Array to keep the changes which will be passed in the df_all_batches DataFrame.

    for n in range(1, iterations_number+1):
        if(verbosity > 1):
            print("Iteration ", n, "---------------------------------------------------------------")

        timeInitial = time.time_ns() # Initial time


        if( n % batch_size ==0 ):
            
            if(verbosity >1):
                print(n,"iterations reached, time to store",batch_size,"draws.")
            draws = np.append(draws, np.array([np.append(draw[:-1], multiplicity)]), axis=0)

            if(verbosity > 1):
                print("The last draw", draw[:-1],"has been stored with multiplicity", draw[-1])
            
            acceptance_percentage = acceptance_ratio * 100/ batch_size

            #print(acceptance_percentage,"of proposed draws have beem accepted in this batch (in master chain).")

            batch_counter += 1

            # Create file and layout.
            if(not os.path.exists(str(draws_folder)+"/draws_batch_n_"+str(batch_counter)+".csv")):
                os.makedirs(str(draws_folder), exist_ok=True)   
            
            
            df_current_batch = pd.DataFrame(
                draws,
                columns=[f"dimension{i+1}" for i in range(dimensions)] + ["multiplicity"]
            ) # DataFrame to save the states visited in the current batch.
            
            # Write new batch to csv.
            df_current_batch.to_csv(str(draws_folder)+"/draws_batch_n_"+str(batch_counter)+".csv", index=False, mode='w', header=True)

            
            time_final = time.time_ns()

            all_batches_statistics = np.append(all_batches_statistics, [[acceptance_percentage,  round((time_final - timeInitial * 10**(-9) ), ndigits=4)]], axis=0)

            # Reinitialize the data for the next batch.
            acceptance_ratio = 0
            multiplicity = 1

            draws = np.zeros(shape=(0,dimensions + 1 ), dtype=np.int64)

            # Save results after the last batch.
            if(n == iterations_number):
                 # Add batches' attributes to dataframe.
                df_all_batches = pd.DataFrame(all_batches_statistics, columns=[ "acceptance rate (%)", "run time (s)"])
                # Write them to csv. 
                df_all_batches.to_csv(str(draws_folder)+"/statistics_batches.csv", index=False, mode='w', header=True)


        RW_monitor = True # Check whether the proposed state is the same as the current.

        # Sampling proposed draw.
        conditional_current_probability = conditional_proposed_probability = 1.0

        for i in range(len(gaussian_draw)):

            while(True):
                # draw_float_sample = np.repeat(2*grid_length, dimensions) # To sample the Guassian deformation.
                # while(np.any(draw_float_sample >= grid_length)):
                #     draw_float_sample = np.around(deviation*np.random.randn(dimensions) + mean, decimals=0).astype(int)
                draw_float_sample = np.around(deviation*np.random.randn() + mean, decimals=0).astype(int)
                                                
                # gaussian_draw[i] = draw_float_sample[i] # Return an array of integers from a normal distribution.
                gaussian_draw[i] = draw_float_sample
                proposed_draw[i] = draw[i] + gaussian_draw[i] # Find the proposed draw.
            
                if((0 <= proposed_draw[i]) and (proposed_draw[i] < grid_length) ): # Conditions to respect in order to continue with the proposed state.
                    break
            
            if(gaussian_draw[i] != 0):
                RW_monitor = False
            
            conditional_current_probability *= truncated_coefficients[draw[i]] # Conditional probability of the current draw.
            conditional_proposed_probability *= truncated_coefficients[proposed_draw[i]] # Conditional probability of the proposed state.

        if(RW_monitor == True):
            acceptance_ratio += 1

            multiplicity += 1

            if(verbosity > 1):
                print("The propposed_draw:", proposed_draw[-1], "turns out to be equal to the current draw:", draw[:-1],"\nso that the multiplicity of the current draw is raised to",multiplicity)
            
        else:
            if(verbosity > 1):
                print("draw is",draw[:-1],"\nproposed_draw is",proposed_draw,"\nprob is",prob)
            
            
                    
            proposed_prob = position_probability[tuple(proposed_draw)]


            probability = np.amin([1, (proposed_prob/prob) * (conditional_current_probability / conditional_proposed_probability)]) # The stohastic part of the MCMC.

            if(np.isnan(probability)):
                raise ValueError("_got NaN while computing densitites ration: proposed_draw", proposed_draw, "prob= ", prob)

            random_number = np.random.rand()

            if(random_number < probability):
                if(verbosity > 1):
                    print("proposed_draw", proposed_draw, "was accepted, since p=", probability, "and random_number", random_number)

                if(n > burn_factor):
                    
                    draws = np.append(draws, np.array([np.append(draw[:-1], multiplicity)]), axis=0) # Add state to the batch's draws.

                    if(verbosity > 1):
                        print("The old draw", draw[:-1], "has been stores with multiplicity", draw[-1])
                    
                multiplicity = 1

                for i in range(len(proposed_draw)):
                    draw[i] = proposed_draw[i] # Set proposed state as the current state.
                
                prob = proposed_prob
                acceptance_ratio += 1

                if(verbosity > 1):
                    print("Now the new draw is", draw[:-1],"\nthe new prob is", prob)
            
            else:
                multiplicity += 1
                if(verbosity > 1):
                    print("proposed_draw", proposed_draw, "was rejected, since probability=", probability, "and random_number=",random_number)
                    print("The current draw", draw[:-1], "remains the same and its multiplicity is:", multiplicity)
                    print("Prob", prob, "remains the same")
            



    