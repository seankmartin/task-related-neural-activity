import matplotlib.pyplot as plt
import numpy as np


def simple_trajectory_plot(correct, incorrect):
    """
    Plot the average trajectory for correct and incorrect trials.

    Parameters:
    -----------
    correct: np.ndarray
        The trajectories of the neurons with GPFA applied for correct trials.
    incorrect: np.ndarray
        The trajectories of the neurons with GPFA applied for incorrect trials.

    Returns:
    --------
    figure: matplotlib.figure.Figure

    """
    average_trajectory_pass = np.mean(correct, axis=0)
    average_trajectory_fail = np.mean(incorrect, axis=0)

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")

    # Do the plot for pass and fail
    ax.plot(*average_trajectory_pass, label="Catch neural trajectory")
    ax.plot(
        average_trajectory_pass[0][0],
        average_trajectory_pass[1][0],
        average_trajectory_pass[2][0],
        "o",
        color="green",
        label="Start",
    )
    ax.plot(
        average_trajectory_pass[0][-1],
        average_trajectory_pass[1][-1],
        average_trajectory_pass[2][-1],
        "o",
        color="red",
        label="End",
    )
    ax.plot(*average_trajectory_fail, "--", label="Miss nueral trajectory")
    ax.plot(
        average_trajectory_fail[0][0],
        average_trajectory_fail[1][0],
        average_trajectory_fail[2][0],
        "o",
        color="green",
        label="Start",
    )
    ax.plot(
        average_trajectory_fail[0][-1],
        average_trajectory_fail[1][-1],
        average_trajectory_fail[2][-1],
        "o",
        color="red",
        label="End",
    )
    ax.legend()

    return fig
