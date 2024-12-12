#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
#SB3 Imports
from stable_baselines3.common.results_plotter import load_results, ts2xy

#A function to smooth the model performance graph by taking a moving average of performance.
def moving_average(values, window, gaussian = False):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    #We create the vector to multiply each value by to get the moving average. Essentially a vector of length n
    # in which each weight is 1/n.
    kernel = np.repeat(1.0, window) / window
    if (gaussian == True) :
        if window % 2 == 0:
            window+=1
        x = np.arange(-(window // 2), window // 2 + 1)
        kernel = np.exp(-(x ** 2) / (2 * window ** 2))
        kernel = kernel / np.sum(kernel)
    #The convolve function iteratively multiplies the first n values in the values array by the weights array.
    # with the given weights array, it essentially takes the moving average of each N values in the values array.
    return np.convolve(values, kernel, "valid")

def plot_results(log_folder, start = 0, stop = None, window = 1, gaussian = False, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the resFults to plot
    :param title: (str) the title of the task to plot
    :param gaussian: (bool) whether to use a flat or gaussian convolution kernel
    """

    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window = window, gaussian = gaussian)
    
    # Truncate x
    if stop is None:
        stop = len(x)
    x = x[(len(x) - len(y)) + start : stop]
    
    # Plot
    plt.plot(x,y)
    plt.xlabel("# of Training Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(title + f"Smoothed, window-size = {window}")
    plt.show()

def plot_multi(log_folders, labels, start = 0, stop = None, window = 1, gaussian = False, title="Learning Curve "):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    for i, log_folder in enumerate(log_folders):
        x, y = ts2xy(load_results(log_folder), "timesteps")
        y = moving_average(y, window = window, gaussian = gaussian)
        
        # Truncate x
        if stop is None:
            stop = len(x)
        x = x[(len(x) - len(y)) + start : stop if stop < len(x) else len(x)]
        plt.plot(x,y, label = labels[i])

    plt.xlabel("# of Training Timesteps")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.title(title + f"Smoothed, window-size = {window}")
    plt.show()