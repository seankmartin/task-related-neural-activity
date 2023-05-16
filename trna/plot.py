import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr
from matplotlib.lines import Line2D
from trna.distance_measurement import procrustes_modify, distance_between_curves


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

    fig = plot_curves(average_trajectory_pass, average_trajectory_fail)
    return fig


def plot_all_trajectories(correct, incorrect, elev=25, azim=-45):
    """
    Plot all the trajectories for correct and incorrect trials.

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
    smr.set_plot_style()
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    ax.set_ylabel("z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for i in range(correct.shape[0])[:5]:
        ax.plot(*correct[i], color="green", alpha=0.3, lw=1)
    for i in range(incorrect.shape[0])[:5]:
        ax.plot(*incorrect[i], color="red", alpha=0.3, lw=1)
    ax.view_init(elev=elev, azim=azim)

    custom_lines = [
        Line2D([0], [0], color="green", alpha=0.3, lw=1),
        Line2D([0], [0], color="red", alpha=0.3, lw=1),
    ]

    ax.legend(
        custom_lines,
        ["Correct trials", "Incorrect trials"],
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        borderaxespad=0.0,
    )
    smr.despine()
    return fig


def plot_curves(
    average_trajectory_pass,
    average_trajectory_fail,
    elev=25,
    azim=-45,
    ax=None,
    do_legend=True,
):
    smr.set_plot_style()
    if ax is None:
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()
    ax.set_ylabel("z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Do the plot for pass and fail
    ax.plot(
        *average_trajectory_pass, label="Catch neural trajectory", marker="o", lw=3.0
    )
    ax.plot(
        *average_trajectory_fail, label="Miss neural trajectory", marker="o", lw=3.0
    )
    ax.plot(
        average_trajectory_pass[0][0],
        average_trajectory_pass[1][0],
        average_trajectory_pass[2][0],
        "o",
        color="green",
        label="Start (0.0s)",
    )
    ax.plot(
        average_trajectory_pass[0][-1],
        average_trajectory_pass[1][-1],
        average_trajectory_pass[2][-1],
        "o",
        color="red",
        label="End (1.0s)",
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
    if do_legend:
        ax.legend(bbox_to_anchor=(1.08, 1), loc="upper left", borderaxespad=0.0)
    ax.view_init(elev=elev, azim=azim)
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
        item = tu["elephant"]
        correct, incorrect = item["correct"], item["incorrect"]
        average_trajectory_pass = np.mean(correct, axis=0)
        average_trajectory_fail = np.mean(incorrect, axis=0)

        original_distance = distance_between_curves(
            average_trajectory_pass, average_trajectory_fail
        )

        ortho_procrustes = procrustes_modify(
            average_trajectory_pass, average_trajectory_fail, orthogonal=True
        )
        ortho_distance = distance_between_curves(*ortho_procrustes)

        procrustes = procrustes_modify(
            average_trajectory_pass, average_trajectory_fail, orthogonal=False
        )
        distance = distance_between_curves(*procrustes)

        angles = [{"elev": 30, "azim": 70}, {"elev": 50, "azim": -45}]
        for angle in angles:
            fig = plt.figure(figsize=plt.figaspect(0.33))
            ax = fig.add_subplot(1, 3, 1, projection="3d")
            ax.set_title(f"Average trajectory - {original_distance:.2f}")
            plot_curves(
                average_trajectory_pass,
                average_trajectory_fail,
                ax=ax,
                **angle,
                do_legend=False,
            )
            ax = fig.add_subplot(1, 3, 2, projection="3d")
            ax.set_title(f"Orthogonal Procrustes - {ortho_distance:.2f}")
            plot_curves(*ortho_procrustes, ax=ax, **angle, do_legend=False)
            ax = fig.add_subplot(1, 3, 3, projection="3d")
            ax.set_title(f"Procrustes - {distance:.2f}")
            plot_curves(*procrustes, ax=ax, **angle)
            angle_as_name = f"{angle['elev']}_{angle['azim']}"
            filename = (
                out_dir
                / f"{tu['name']}_gpfa_curves_{angle_as_name}_{brain_regions}_{t}.png"
            )
            fig = smr.SimuranFigure(fig, filename=filename)
            fig.save()

        starting_distance = np.linalg.norm(
            average_trajectory_pass[:, 0] - average_trajectory_fail[:, 0]
        )
        ending_distance = np.linalg.norm(
            average_trajectory_pass[:, -1] - average_trajectory_fail[:, -1]
        )
        correct_variance = np.mean(np.var(correct, axis=0))
        incorrect_variance = np.mean(np.var(incorrect, axis=0))
        list_info.append(
            [
                ortho_distance,
                distance,
                original_distance,
                starting_distance,
                ending_distance,
            ]
        )
        l2_info.append([correct_variance, "Correct"])
        l2_info.append([incorrect_variance, "Incorrect"])

    df = pd.DataFrame(
        list_info,
        columns=[
            "Orthogonal procrustes distance",
            "Procrustes distance",
            "Trajectory distance",
            "Start distance",
            "End distance",
        ],
    )
    df2 = pd.DataFrame(l2_info, columns=["Variance", "Trial result"])
    filename = str(out_dir / f"gpfa_distance_{brain_regions}_{t}.csv")
    df.to_csv(filename, index=False)
    filename = str(out_dir / f"gpfa_variance_{brain_regions}_{t}.csv")
    df2.to_csv(filename, index=False)
    smr.set_plot_style()

    for x in ["Procrustes distance", "Start distance", "End distance"]:
        fig, ax = plt.subplots()
        sns.histplot(df, x=x, ax=ax, kde=True)
        filename = str(
            out_dir
            / f"gpfa_distance_{x.lower().replace(' ', '_')}_{brain_regions}_{t}.png"
        )
        smr_fig = smr.SimuranFigure(fig, filename)
        smr.despine()
        smr_fig.save()

    fig, ax = plt.subplots()
    sns.histplot(df2, x="Variance", hue="Trial result", kde=True, ax=ax)
    smr.despine()
    filename = str(out_dir / f"gpfa_variance_{brain_regions}_{t}.png")
    fig = smr.SimuranFigure(fig, filename)
    fig.save()


def plot_cca_correlation(recording_info, out_dir, n, regions):
    list_info = []
    for tu in recording_info:
        correct, incorrect = tu["scikit"]["correct"], tu["scikit"]["incorrect"]
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
    df.to_csv(out_dir / f"cca_correlation{n}_{regions}.csv", index=False)

    smr.set_plot_style()
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
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
            list_info.append([np.mean(val1), np.mean(val2), "Correct"])
    for trial in incorrect:
        delay = trial[0]
        if delay != 0:
            continue
        cca = trial[1]
        for val1, val2 in zip(cca[0], cca[1]):
            list_info.append([np.mean(val1), np.mean(val2), "Incorrect"])
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
        x=f"{br1} canonical dimension",
        y=f"{br2} canonical dimension",
        hue="Trial result",
        style="Trial result",
        ax=ax,
    )

    smr.despine()
    return fig


def plot_cca_example(recording_info, brain_regions, t=0):
    list_info = []
    p1_info = []
    p2_info = []
    info = recording_info["scikit"]
    correct, incorrect = info["correct"], info["incorrect"]
    for i, trial in enumerate(correct):
        delay = trial[0]
        cca = trial[1]
        list_info.append([delay, i, np.mean(cca[0]), np.mean(cca[1]), "Correct"])
        original = trial[2]
        p1_info.append(
            [delay, i, np.mean(original[0][0]), np.mean(original[0][1]), "Correct"]
        )
        p2_info.append(
            [delay, i, np.mean(original[1][0]), np.mean(original[1][1]), "Correct"]
        )

    for trial in incorrect:
        delay = trial[0]
        cca = trial[1]
        list_info.append([delay, i, np.mean(cca[0]), np.mean(cca[1]), "Incorrect"])
        original = trial[2]
        p1_info.append(
            [delay, i, np.mean(original[0][0]), np.mean(original[0][1]), "Incorrect"]
        )
        p2_info.append(
            [delay, i, np.mean(original[1][0]), np.mean(original[1][1]), "Incorrect"]
        )

    region1 = brain_regions[0]
    if isinstance(region1, list):
        region1 = region1[0][0]
    region2 = brain_regions[1]
    df1 = pd.DataFrame(
        list_info,
        columns=[
            "Delay",
            "Trial number",
            f"{region1} canonical dimension",
            f"{region2} canonical dimension",
            "Trial result",
        ],
    )
    df2 = pd.DataFrame(
        p1_info,
        columns=[
            "Delay",
            "Trial number",
            f"{region1} neuron 1",
            f"{region1} neuron 2",
            "Trial result",
        ],
    )
    df3 = pd.DataFrame(
        p2_info,
        columns=[
            "Delay",
            "Trial number",
            f"{region2} neuron 1",
            f"{region2} neuron 2",
            "Trial result",
        ],
    )

    smr.set_plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.scatterplot(
        df1[(df1["Delay"] == t)][:30],
        x=f"{region1} canonical dimension",
        y=f"{region2} canonical dimension",
        hue="Trial result",
        style="Trial result",
        ax=ax[0],
    )
    sns.scatterplot(
        df2[(df2["Delay"] == t)][:30],
        x=f"{region1} neuron 1",
        y=f"{region1} neuron 2",
        hue="Trial result",
        style="Trial result",
        ax=ax[1],
    )
    sns.scatterplot(
        df3[(df3["Delay"] == t)][:30],
        x=f"{region2} neuron 1",
        y=f"{region2} neuron 2",
        hue="Trial result",
        style="Trial result",
        ax=ax[2],
    )

    smr.despine()
    return fig
