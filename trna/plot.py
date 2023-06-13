import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr
from matplotlib.lines import Line2D
from trna.distance_measurement import procrustes_modify, distance_between_curves
from trna.common import scale_data


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


def plot_all_trajectories(correct, incorrect, elev=25, azim=-45, num=200):
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
    ax.set_zlabel("z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    num1 = min(num, correct.shape[0])
    num2 = min(num, incorrect.shape[0])
    for i in range(num1):
        ax.plot(*correct[i], color="darkblue", alpha=0.3, lw=1)
    for i in range(num2):
        ax.plot(*incorrect[i], color="sienna", alpha=0.3, lw=1)
    ax.view_init(elev=elev, azim=azim)

    custom_lines = [
        Line2D([0], [0], color="darkblue", alpha=0.3, lw=1),
        Line2D([0], [0], color="sienna", alpha=0.3, lw=1),
    ]

    ax.legend(
        custom_lines,
        ["Catch trials", "Miss trials"],
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        borderaxespad=0.0,
    )
    smr.despine()
    return fig


def plot_trajectories_split(correct, incorrect, elev=25, azim=-45, num=200):
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
    smr.set_plot_style()
    fig = plt.figure(figsize=plt.figaspect(1.5))
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    ax.set_zlabel("z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Catch trials")
    ax.view_init(elev=elev, azim=azim)
    num1 = min(num, correct.shape[0])
    num2 = min(num, incorrect.shape[0])
    for i in range(num1):
        ax.plot(*correct[i], color="darkblue", alpha=0.3, lw=1)
    smr.despine()

    ax = fig.add_subplot(2, 1, 2, projection="3d")
    ax.set_zlabel("z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Miss trials")
    ax.view_init(elev=elev, azim=azim)
    for i in range(num2):
        ax.plot(*incorrect[i], color="sienna", alpha=0.3, lw=1)

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
    ax.set_zlabel("z")
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
        color="seagreen",
        label="Start (0.0s)",
    )
    ax.plot(
        average_trajectory_pass[0][-1],
        average_trajectory_pass[1][-1],
        average_trajectory_pass[2][-1],
        "o",
        color="maroon",
        label="End (1.0s)",
    )
    ax.plot(
        average_trajectory_fail[0][0],
        average_trajectory_fail[1][0],
        average_trajectory_fail[2][0],
        "o",
        color="seagreen",
    )
    ax.plot(
        average_trajectory_fail[0][-1],
        average_trajectory_fail[1][-1],
        average_trajectory_fail[2][-1],
        "o",
        color="maroon",
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
        correct, incorrect = tu["correct"], tu["incorrect"]
        correct, incorrect = scale_data(correct, incorrect, uniform=True)

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
                / brain_regions
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
                (starting_distance + ending_distance) / 2,
                correct_variance + incorrect_variance / 2,
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
            "Average distance",
            "Average variance",
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
    sns.regplot(df, y="Average distance", x="Procrustes distance", ax=ax)
    filename = str(out_dir / f"gpfa_distance_average_{brain_regions}_{t}.png")
    smr_fig = smr.SimuranFigure(fig, filename)
    smr.despine()
    smr_fig.save()

    fig, ax = plt.subplots()
    sns.regplot(df, x="Procrustes distance", y="Average variance", ax=ax)
    filename = str(out_dir / f"gpfa_procrustes_vs_variance_{brain_regions}_{t}.png")
    smr_fig = smr.SimuranFigure(fig, filename)
    smr.despine()
    smr_fig.save()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = df["Average distance"]
    y = df["Procrustes distance"]
    z = df["Average variance"]
    ax.plot(x, y, z, "o")

    ax.plot(x, y, ".", zdir="z", zs=-0.2, color="black")
    ax.plot(x, z, ".", zdir="y", zs=-0.2, color="black")
    ax.plot(y, z, ".", zdir="x", zs=-0.2, color="black")

    ax.set_xlabel("Average distance")
    ax.set_ylabel("Procrustes distance")
    ax.set_zlabel("Average variance")
    filename = str(out_dir / f"gpfa_3dscatter_{brain_regions}_{t}.png")
    smr.despine()
    smr_fig = smr.SimuranFigure(fig, filename)
    smr_fig.save()

    fig, ax = plt.subplots()
    sns.scatterplot(
        df, x="Average distance", y="Procrustes distance", hue="Average variance", ax=ax
    )
    filename = str(out_dir / f"gpfa_distance_average_variance_{brain_regions}_{t}.png")
    smr.despine()
    smr_fig = smr.SimuranFigure(fig, filename)
    smr_fig.save()

    fig, ax = plt.subplots()
    sns.histplot(df2, x="Variance", hue="Trial result", kde=True, ax=ax)
    smr.despine()
    filename = str(out_dir / f"gpfa_variance_{brain_regions}_{t}.png")
    fig = smr.SimuranFigure(fig, filename)
    fig.save()


def plot_cca_correlation(recording_info, out_dir, n, regions, scatter=True):
    if len(recording_info) == 0:
        return {}
    list_info = []
    for tu in recording_info:
        correct = tu["unsplit_corr_info_correct"]
        incorrect = tu["unsplit_corr_info_incorrect"]
        for trial in correct:
            list_info.append([*trial, "Correct"])
        for trial in incorrect:
            list_info.append([*trial, "Incorrect"])
    df = pd.DataFrame(
        list_info, columns=["Delay", "Population correlation", "Trial result"]
    )
    df.to_csv(out_dir / f"cca_correlation_{n}_{regions}.csv", index=False)

    list_info = []
    for tu in recording_info:
        correct_rates, incorrect_rates = tu["correct_rates"], tu["incorrect_rates"]
        for trial in correct_rates:
            delay = trial[0]
            rate = trial[1]
            per_neuron_rate = trial[2]
            list_info.append(
                [
                    delay,
                    rate[0],
                    rate[1],
                    np.mean(per_neuron_rate[0]),
                    np.mean(per_neuron_rate[1]),
                    "Correct",
                ]
            )
        for trial in incorrect_rates:
            delay = trial[0]
            rate = trial[1]
            per_neuron_rate = trial[2]
            list_info.append(
                [
                    delay,
                    rate[0],
                    rate[1],
                    np.mean(per_neuron_rate[0]),
                    np.mean(per_neuron_rate[1]),
                    "Incorrect",
                ]
            )
    df2 = pd.DataFrame(
        list_info,
        columns=[
            "Delay",
            "Rate 1 reduced",
            "Rate 2 reduced",
            "Average rate 1",
            "Average rate 2",
            "Trial result",
        ],
    )
    df2.to_csv(out_dir / f"rate_correlation_{n}_{regions}.csv", index=False)

    list_info = []
    for tu in recording_info:
        correct = tu["concat_correct_corr_info"]
        incorrect = tu["concat_incorrect_corr_info"]
        for trial in correct:
            list_info.append([*trial, "Correct"])
        for trial in incorrect:
            list_info.append([*trial, "Incorrect"])
    df3 = pd.DataFrame(
        list_info, columns=["Delay", "Population correlation", "Trial result"]
    )
    df3.to_csv(out_dir / f"concat_cca_correlation_{n}_{regions}.csv", index=False)

    smr.set_plot_style()

    if len(df[df["Delay"] == 0]) == 0:
        fig1 = None
    else:
        fig1, ax = plt.subplots()
        sns.lineplot(
            data=df,
            x="Delay",
            y="Population correlation",
            hue="Trial result",
            style="Trial result",
            ax=ax,
        )
        ax.set_title("CCA correlation per trial")
        smr.despine()

    if len(df2[df2["Delay"] == 0]) == 0:
        fig2 = None
    else:
        fig2 = sns.lmplot(
            data=df2[df2["Delay"] == 0],
            x="Average rate 1",
            y="Average rate 2",
            hue="Trial result",
            markers=["o", "x"],
            scatter=scatter,
        )
        smr.despine()

    if len(df3[df3["Delay"] == 0]) == 0:
        fig3 = None
    else:
        fig3, ax = plt.subplots()
        sns.lineplot(
            data=df3,
            x="Delay",
            y="Population correlation",
            hue="Trial result",
            style="Trial result",
            ax=ax,
        )
        ax.set_title(f"CCA correlation for {regions.replace('_', ' ')}")
        smr.despine()

    return {
        "per_trial": fig1,
        "rate_based": fig2,
        "concat": fig3,
    }


def plot_cca_example(recording_info, brain_regions, t=0, num=10, num2=10, win_len=1):
    list_info = []
    p1_correct = []
    p2_incorrect = []
    p1_incorrect = []
    p2_correct = []
    info = recording_info
    correct, incorrect = info["correct_rates"], info["incorrect_rates"]
    per_trial_spikes = info["per_trial_spikes"]
    indices_to_get1 = []
    indices_to_get2 = []
    for i, trial in enumerate(correct):
        region1_rate, region2_rate = trial[2]
        for j, val in enumerate(region1_rate):
            if val > 0 and len(indices_to_get1) < 3:
                indices_to_get1.append(j)
        for j, val in enumerate(region2_rate):
            if val > 0 and len(indices_to_get2) < 3:
                indices_to_get2.append(j)

    for i, trial in enumerate(correct):
        delay = trial[0]
        cca = trial[1]
        list_info.append([delay, i, cca[0], cca[1], "Correct"])
        region1_rate, region2_rate = trial[2]
        if delay == t:
            p1_correct.append(region1_rate[indices_to_get1])
            p2_correct.append(region2_rate[indices_to_get2])

    for i, trial in enumerate(incorrect):
        delay = trial[0]
        cca = trial[1]
        list_info.append([delay, i, cca[0], cca[1], "Incorrect"])
        region1_rate, region2_rate = trial[2]
        if delay == t:
            p1_incorrect.append(region1_rate[indices_to_get1])
            p2_incorrect.append(region2_rate[indices_to_get2])

    region1 = brain_regions[0]
    if isinstance(region1, list):
        region1 = region1[0][0]
    region2 = brain_regions[1]
    if isinstance(region2, list):
        region2 = region2[0][0]

    smr.set_plot_style()
    # Plot 1 and 2 - region1 and region2 average rates in 3d
    fig0 = plt.figure()
    ax = fig0.add_subplot(projection="3d")
    ax.set_title(f"{region1} average rates")
    for p in p1_correct[:num]:
        ax.plot(*p, label=f"Correct", c="sienna", marker="o")
    for p in p1_incorrect[:num]:
        ax.plot(*p, label=f"Incorrect", c="darkblue", marker="x")
    ax.set_xlabel("N1")
    ax.set_ylabel("N2")
    ax.set_zlabel("N3")
    custom_lines = [
        Line2D([], [], color="sienna", alpha=0.8, linestyle="none", marker="o"),
        Line2D([], [], color="darkblue", alpha=0.8, linestyle="none", marker="x"),
    ]

    ax.legend(
        custom_lines,
        ["Correct trials", "Incorrect trials"],
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        borderaxespad=0.0,
    )
    smr.despine()

    fig1 = plt.figure()
    ax = fig1.add_subplot(projection="3d")
    ax.set_title(f"{region2} average rates")
    for p in p2_correct[:num]:
        ax.plot(*p, label=f"Correct", c="sienna", marker="o")
    for p in p2_incorrect[:num]:
        ax.plot(*p, label=f"Incorrect", c="darkblue", marker="x")
    ax.set_xlabel("N1")
    ax.set_ylabel("N2")
    ax.set_zlabel("N3")
    custom_lines = [
        Line2D([], [], color="sienna", alpha=0.8, linestyle="none", marker="o"),
        Line2D([], [], color="darkblue", alpha=0.8, linestyle="none", marker="x"),
    ]

    ax.legend(
        custom_lines,
        ["Correct trials", "Incorrect trials"],
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        borderaxespad=0.0,
    )
    smr.despine()

    # Plot 3 - spike rates in 2d
    fig2, ax = plt.subplots()
    num_raster_trials = num2
    for i, region in enumerate(per_trial_spikes):
        for j, trial in enumerate(region[:num_raster_trials]):
            change = i * len(per_trial_spikes[0][0])
            for k, st in enumerate(trial):
                st_to_plot = np.array(st) + (j * win_len)
                ax.scatter(
                    st_to_plot,
                    k * np.ones_like(st_to_plot) + change,
                    s=0.1,
                    c="k",
                    marker=".",
                )
            if j != 0:
                ax.vlines(
                    win_len * j,
                    change,
                    change + len(trial),
                    color="r",
                    linestyle="--",
                    linewidths=0.5,
                )
    ax.hlines(
        len(per_trial_spikes[0][0]),
        0,
        win_len * num2,
        color="r",
        linestyle="--",
        linewidths=0.5,
    )
    ax.set_title("Spike trains")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike index")
    ax.invert_yaxis()
    smr.despine()

    # Plot 4 - cca in 2d of the two regions
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

    fig3, ax = plt.subplots()
    correct = df1[(df1["Delay"] == t) & (df1["Trial result"] == "Correct")][:num]
    incorrect = df1[(df1["Delay"] == t) & (df1["Trial result"] == "Incorrect")][:num]
    ax.scatter(
        correct[f"{region1} canonical dimension"],
        correct[f"{region2} canonical dimension"],
        c="sienna",
        marker="o",
        label="Correct",
    )
    ax.scatter(
        incorrect[f"{region1} canonical dimension"],
        incorrect[f"{region2} canonical dimension"],
        c="darkblue",
        marker="x",
        label="Incorrect",
    )
    ax.set_title(f"CCA of {region1} and {region2}")
    ax.set_xlabel(f"{region1} canonical dimension")
    ax.set_ylabel(f"{region2} canonical dimension")
    ax.legend()

    smr.despine()

    return {f"{region1}": fig0, f"{region2}": fig1, "spike_train": fig2, "cca": fig3}
