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

def plot_results(log_folder, x_range = None, smoothing = True, window = 100, gaussian = True, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the resFults to plot
    :param x_range: (str) the range of the plot on the X-axis
    :param smoothing: (bool) whether to use smoothing on the learning curve
    :param window: (int) the size of the smoothing kernel
    :param gaussian: (bool) whether to use a gaussian or flat smoothing convolution kernel
    :param title: (str) the title of the task to plot
    """

    x, y = ts2xy(load_results(log_folder), "timesteps")
    if smoothing is True:
        y = moving_average(y, window = window, gaussian = gaussian)
    x = x[(len(x) - len(y)):]
    # Plot
    plt.plot(np.arange(len(y)),y)
    plt.xlabel("# of Training Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(title + f"Smoothed, window-size = {window}")
    plt.show()

def plot_multi(log_folders, labels, x_range = None, smoothing = True, window = 100, gaussian = False, title="Learning Curve "):
    """
    plot the results

    :param log_folders: (str) the save locations of the results for each line in the multi-line plot
    :param labels: The labels of each line in the multi-line plot.
    :param x_range: (str) the range of the plot on the X-axis
    :param smoothing: (bool) whether to use smoothing on the learning curve
    :param window: (int) the size of the smoothing kernel
    :param gaussian: (bool) whether to use a gaussian or flat smoothing convolution kernel
    :param title: (str) the title of the task to plot
    """
    for i, log_folder in enumerate(log_folders):
        x, y = ts2xy(load_results(log_folder), "timesteps")
        if smoothing is True:
            y = moving_average(y, window = window, gaussian = gaussian)
        # Truncate x
        x = x[(len(x) - len(y)):]
        plt.plot(x,y, label = labels[i])
    if x_range is not None:
        plt.xlim(x_range[0],x_range[1])
    plt.xlabel("# of Training Timesteps")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.title(title + f"Smoothed, window-size = {window}")
    plt.show()