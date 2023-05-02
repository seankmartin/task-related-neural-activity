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


def plot_gpfa_distance(recording_info, out_dir, brain_regions, t):
    """
    Plot the distance between the average trajectory for correct and incorrect
    trials.

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
    list_info = []
    l2_info = []
    for tu in recording_info:
        correct, incorrect = tu["elephant"]
        average_trajectory_pass = np.mean(correct, axis=0)
        average_trajectory_fail = np.mean(incorrect, axis=0)

        distance = np.linalg.norm(
            average_trajectory_pass - average_trajectory_fail, axis=0
        )
        starting_distance = np.linalg.norm(
            average_trajectory_pass[:, 0] - average_trajectory_fail[:, 0]
        )
        ending_distance = np.linalg.norm(
            average_trajectory_pass[:, -1] - average_trajectory_fail[:, -1]
        )
        correct_variance = np.var(correct, axis=0)
        incorrect_variance = np.var(incorrect, axis=0)
        list_info.append(
            [
                distance,
                starting_distance,
                ending_distance,
            ]
        )
        l2_info.append([correct_variance, "Correct"])
        l2_info.append([incorrect_variance, "Incorrect"])

    df = pd.DataFrame(
        list_info,
        columns=[
            "Trajectory distance",
            "Start distance",
            "End distance",
        ],
    )
    df2 = pd.DataFrame(l2_info, columns=["Variance", "Trial result"])
    smr.set_plot_style()

    for x in ["Trajectory distance", "Start distance", "End distance"]:
        fig, ax = plt.subplots()
        sns.histplot(df, x=x, ax=ax, kde=True)
        filename = str(
            out_dir
            / f"gpfa_distance_{x.lower().replace(' ', '_')}_{brain_regions}_{t}.png"
        )
        smr_fig = smr.SimuranFigure(fig, filename)
        smr_fig.save()
        smr.despine()

    fig, ax = plt.subplots()
    sns.histplot(df2, x="Variance", hue="Trial result", kde=True, ax=ax)
    smr.despine()
    filename = str(out_dir / f"gpfa_variance_{brain_regions}_{t}.png")
    fig = smr.SimuranFigure(fig, filename)
    fig.save()


def plot_cca_correlation(recording_info):
    list_info = []
    for tu in recording_info:
        correct, incorrect = tu["scikit"]
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


def plot_cca_correlation_features(correct, incorrect, brain_regions):
    br1, br2 = brain_regions
    if isinstance(br2, list):
        br2 = br2[0][:-1]
    list_info = []
    for trial in correct:
        delay = trial[0]
        if delay != 0:
            continue
        cca = trial[1]
        for val1, val2 in zip(cca[0], cca[1]):
            list_info.append([val1, val2, "Correct"])
    for trial in incorrect:
        delay = trial[0]
        if delay != 0:
            continue
        cca = trial[1]
        for val1, val2 in zip(cca[0], cca[1]):
            list_info.append([val1, val2, "Incorrect"])
    df = pd.DataFrame(
        list_info,
        columns=[
            f"{br1} canonical dimension",
            f"{br2} canonical dimension",
            "Trial result",
        ],
    )

    smr.set_plot_style()
    fig, ax = plt.subplots()
    sns.scatterplot(
        df,
        x="Delay",
        y="Population correlation",
        hue="Trial result",
        style="Trial result",
        ax=ax,
    )

    smr.despine()
    return fig
