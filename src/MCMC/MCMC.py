import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import truncnorm

os.chdir(sys.path[0])

def grid_rewards_2d(gridLength=13, r0=0.1, r1=0.5, r2=2.0):
    """
    Calculate a 2D grid reward fucntion.
    
    Parameters
    ----------

    gridLength: (float)
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
                - A 2D array with the corresponding values of the function in the 2D grid of length gridLength.
                      
    """

    assert gridLength >= 7
    
    coord_1d_dist_from_center = np.abs(np.arange(gridLength) / (gridLength - 1) - 0.5)
    mid_level_1d = (coord_1d_dist_from_center > 0.25).astype(float)
    mid_level_2d = mid_level_1d[None, :]*mid_level_1d[:, None]
    high_level_1d = ((coord_1d_dist_from_center > 0.3) * (coord_1d_dist_from_center < 0.4)).astype(float)
    high_level_2d = high_level_1d[None, :]*high_level_1d[:, None]

    rewards = r0 + r1*mid_level_2d + r2*high_level_2d
    return  rewards

def cdfNormaDistribution(mean=0., deviation=1., dataPoints=1e5):
    """
    Calculate the cdf of a normal distribution. 
    
    Parameters
    ----------

    var: (float)
                - The point where we want ot find the cdf value.
    
    mean: (float)
                - The location of the normal distribution.

    deviation: (float)
                - The deviation of the noirmal distribution.

    dataPoints: (integer)
                - Number of points drawn from the normal distribution.
                - Number of points returned.

    cdfAccuracy: (integer)
                - Number of points to regard in the cdf.

    Return:
        -------
                - An array containing two subarrays: one with the values
                of the random variable and one with its corresponding probability
                      
    """
    
    data = deviation*np.random.randn(int(dataPoints)) + mean

    # Sort the data in an ascending order.
    x = np.sort(data)

    # Get the values of the y-axis.
    y = np.arange(dataPoints)/ float(dataPoints)

    
    return [x,y]

def discreteNormalDistribution(gridLength, mean=0.0, deviation=1.0):
    """
    Calculate the noraml disctribution of discrete variables. 
    
    Parameters
    ----------

    dataPoints: (integer)
                - Number of points drawn from the normal distribution.
                - Number of points returned.


    mean: (float)
                - The location of the normal distribution.

    deviation: (float)
                - The deviation of the noirmal distribution.

    Return:
        -------
        truncatedCoefficients: (array)
                 - The probability density function of a truncated discrete normal distribution.
                 - Its number is given by the dataPoints. 
                      
    """
    truncatedCoefficients = np.zeros(gridLength, dtype=np.float64)

    for i in range(0,gridLength):
        cdfDifferences = 0.0
        for n in range(-i, gridLength-i):
            
            # Probability debsity function of a normal distribution.
            var, cdf = cdfNormaDistribution(mean, deviation) # var -> random variable, cdf -> the corresponding value for var

            cdfDifferences += cdf[np.argmax(var > n + 0.5)] - cdf[np.argmax(var > n - 0.5)] # Find the cumulative probability in each point.

        truncatedCoefficients[i] = cdfDifferences
        
    return truncatedCoefficients


def VertexMCMC(gridLength, iterationsNumber, batchSize, burnFactor, verbosity, drawsFolder, deviation, mean = 0., rewardFunction = grid_rewards_2d, dimensions = 2):

    """
    Run the MCMC simulation and save results in the corresponding folder. 
    
    Parameters
    ----------
   
    iterationsNumber: (integer)
                - The number of iterations for the MCMC.

    batchSize: (integer)
                - The number of iterations per batch.

    burnFactor: (integer)
                - The number of iterations discarded in each batch due to non-convergence.

    verbosity: (integer)
                - Whether information should be printed.
    
    drawsFolder: (string)
                - The lcoation of the data folder.
    
    deviation: (float)
                - The standard deviation of our Gaussian distribution.

    mean: (float)
                - The mean of our Gaussian distribution.

    rewardFunction: (function)
                - The reward function our MCMC is trying to lean from. 
                - Should return a 'dimensions' times array where is dimension has gridLength.

    dimensions: (integer)
                - The number of dimensions.
         
    """

    # Make sure number of iterations is an integer multiple of the batch size.
    if(iterationsNumber % batchSize != 0):
        raise ValueError("Number of iterations must be a multiple of batch size.")
    
    
    # The truncated coefficients for the discrete gaussian distribution.
    truncatedCoefficients = discreteNormalDistribution(gridLength, 0, deviation)

    draw = np.zeros(dimensions + 1, dtype=np.int64) # The current draw (state).
    gaussianDraw = np.zeros(dimensions , np.int64) # The guassian deformation to be added in the current draw (state).

    amplitude = 0.0 # value of the reward function for the current draw.
  
    while( amplitude == 0 ):
        for i in range(np.size(draw)):
            draw[i] = np.random.randint(gridLength) # Initialize current draw.
        
        positionProbability = rewardFunction(gridLength) 
        amplitude = positionProbability[tuple(draw[:-1])]
        
    draw[-1] = 1 # Initial multiplicity

    if (verbosity > 1):
        print("Initial draw is", draw[:-1],"with amplitude", amplitude)

    # Proposed draw (to be determinned adding the guassian deformation to the current draw).
    proposedDraw = np.zeros(dimensions , dtype=np.int64)

    acceptanceRatio = 0 # To measure the percentage of accepted moves by the simulator. Estimator of the codes optimization.
    multiplicity = 1    # Initial multiplicity counter.
    
    RWMonitor = True

    dfAllBatches = pd.DataFrame({"acceptance rate (%)":[], "run time (s)":[]}) # DataFrame to save universal data for the eah batch.

    batchCounter = 0 # Indicator to be used naming each batch.

    draws = np.empty( shape=(1,dimensions + 1)) # Array to save the states visited in each batch.

    allBatchesStatistics = np.zeros(shape=(0,2)) # Array to keep the changes which will be passed in the dfAllBatches DataFrame.

    for n in range(1, iterationsNumber+1):
        if(verbosity > 1):
            print("Iteration ", n, "---------------------------------------------------------------")

        timeInitial = time.time_ns() # Initial time


        if( n % batchSize ==0 ):
            
            if(verbosity >1):
                print(n,"iterations reached, time to store",batchSize,"draws.")
            draws = np.append(draws, np.array([np.append(draw[:-1] - 1, multiplicity)]), axis=0)

            if(verbosity > 1):
                print("The last draw", draw[:-1],"has been stored with multiplicity", draw[-1])
            
            acceptancePercentage = acceptanceRatio * 100/ batchSize

            #print(acceptancePercentage,"of proposed draws have beem accepted in this batch (in master chain).")

            batchCounter += 1

            # Create file and layout.
            if(not os.path.exists(str(drawsFolder)+"/draws_batch_n_"+str(batchCounter)+".csv")):
                os.makedirs(str(drawsFolder), exist_ok=True)   
            
            
            dfCurrentBatch = pd.DataFrame(
                draws,
                columns=[f"dimension{i+1}" for i in range(dimensions)] + ["multiplicity"]
            ) # DataFrame to save the states visited in the current batch.
            
            # Write new batch to csv.
            dfCurrentBatch.to_csv(str(drawsFolder)+"/draws_batch_n_"+str(batchCounter)+".csv", index=False, mode='w', header=True)

            
            timeFinal = time.time_ns()

            allBatchesStatistics = np.append(allBatchesStatistics, [[acceptancePercentage,  round((timeFinal - timeInitial * 10**(-9) ), ndigits=4)]], axis=0)

            # Reinitialize the data for the next batch.
            acceptanceRatio = 0
            multiplicity = 1

            draws = np.zeros(shape=(0,dimensions + 1 ), dtype=np.int64)

            # Save results after the last batch.
            if(n == iterationsNumber):
                 # Add batches' attributes to dataframe.
                dfAllBatches = pd.DataFrame(allBatchesStatistics, columns=[ "acceptance rate (%)", "run time (s)"])
                # Write them to csv. 
                dfAllBatches.to_csv(str(drawsFolder)+"/statistics_batches.csv", index=False, mode='w', header=True)


        RWMonitor = True # Check whether the proposed state is the same as the current.

        #Sampling proposed draw.
        conditionalCurrentProbability = conditionalProposedProbability = 1.0

        for i in range(len(gaussianDraw)):

            while(True):
                drawFloatSample = np.repeat(2*gridLength, dimensions) # To sample the Guassian deformation.
                while(np.any(drawFloatSample >= gridLength)):
                    drawFloatSample = np.around(np.random.randn(dimensions), decimals=0).astype(int)
                                                
                gaussianDraw[i] = drawFloatSample[i] # Return an array of integers from a normal distribution.
                proposedDraw[i] = draw[i] + gaussianDraw[i] # Find the proposed draw.
            
                if((1 <= proposedDraw[i]) and (proposedDraw[i] < gridLength) and (proposedDraw[i] > 0)): # Conditions to respect in order to continue with the proposed state.
                    break
            
            if(gaussianDraw[i] != 0):
                RWMonitor = False
            
            conditionalCurrentProbability *= truncatedCoefficients[draw[i]] # Conditional probability of the current draw.
            conditionalProposedProbability *= truncatedCoefficients[proposedDraw[i]] # Conditional probability of the proposed state.

        if(RWMonitor == True):
            acceptanceRatio += 1

            multiplicity += 1

            if(verbosity > 1):
                print("The propposedDraw:", proposedDraw[-1], "turns out to be equal to the current draw:", draw[:-1],"\nso that the multiplicity of the current draw is raised to",multiplicity)
            
        else:
            if(verbosity > 1):
                print("draw is",draw[:-1],"\nproposedDraw is",proposedDraw,"\namplitude is",amplitude)
            
            
            positionProbability = rewardFunction(gridLength=gridLength)            
            proposedAmplitude = positionProbability[tuple(proposedDraw)]


            probability = np.amin([1, (proposedAmplitude**2)/ (amplitude**2) * (conditionalCurrentProbability / conditionalProposedProbability)]) # The stohastic part of the MCMC.

            if(np.isnan(probability)):
                raise ValueError("Got NaN while computing densitites ration: proposedDraw", proposedDraw, "amplitude= ", amplitude)

            randomNumber = np.random.rand()

            if(randomNumber < probability):
                if(verbosity > 1):
                    print("prposedDraw", proposedDraw, "was accepted, since p=", probability, "and randomNumber", randomNumber)

                if(n > burnFactor):
                    
                    draws = np.append(draws, np.array([np.append(draw[:-1] - 1, multiplicity)]), axis=0) # Add state tp the batch's draws.

                    if(verbosity > 1):
                        print("The old draw", draw[:-1], "has been stores with multiplicity", draw[-1])
                    
                multiplicity = 1

                for i in range(len(proposedDraw)):
                    draw[i] = proposedDraw[i] # Set proposed state as the current state.
                
                amplitude = proposedAmplitude
                acceptanceRatio += 1

                if(verbosity > 1):
                    print("Now the new draw is", draw[:-1],"\nthe new amplitude is", amplitude)
            
            else:
                multiplicity += 1
                if(verbosity > 1):
                    print("proposedDraw", proposedDraw, "was rejected, since probability=", probability, "and randomNumber=",randomNumber)
                    print("The current draw", draw[:-1], "remains the same and its multiplicity is:", multiplicity)
                    print("Amplitude", amplitude, "remains the same")
            



    