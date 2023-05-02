import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr


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

    smr.set_plot_style()
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")

    # Do the plot for pass and fail
    ax.plot(*average_trajectory_pass, label="Catch neural trajectory")
    ax.plot(*average_trajectory_fail, "--", label="Miss neural trajectory")
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
    ax.plot(
        average_trajectory_fail[0][0],
        average_trajectory_fail[1][0],
        average_trajectory_fail[2][0],
        "o",
        color="green",
    )
    ax.plot(
        average_trajectory_fail[0][-1],
        average_trajectory_fail[1][-1],
        average_trajectory_fail[2][-1],
        "o",
        color="red",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    smr.despine()

    return fig


def plot_cca_correlation(correct, incorrect):
    list_info = []
    for trial in correct:
        delay = trial[0]
        cca = trial[1]
        correlation = np.corrcoef(cca[0], cca[1])[0, 1]
        list_info.append([delay, correlation, "Correct"])
    for trial in incorrect:
        delay = trial[0]
        cca = trial[1]
        correlation = np.corrcoef(cca[0], cca[1])[0, 1]
        list_info.append([delay, correlation, "Incorrect"])
    df = pd.DataFrame(
        list_info, columns=["Delay", "Population correlation", "Trial result"]
    )

    smr.set_plot_style()
    fig, ax = plt.subplots()
    sns.lineplot(
        df,
        x="Delay",
        y="Population correlation",
        hue="Trial result",
        style="Trial result",
        ax=ax,
    )

    smr.despine()
    return fig
