from trna.common import load_config, perform_scaling_and_stats
from trna.allen import load_allen
from trna.gpfa import analyse_single_recording
from pathlib import Path
from simuran.plot.figure import SimuranFigure
from trna.plot import (
    simple_trajectory_plot,
    plot_trajectories_split,
)
import pandas as pd

# Load the config file
config_path = r"E:\Repos\task-related-neural-activity\config\config.yaml"
config = load_config(config_path=config_path)
out_dir = Path(r"E:\Repos\task-related-neural-activity\workflow\figures\scaling")

rc, loader = load_allen(
    config["allen_data_dir"], config["allen_manifest"], ["CA1", "SUB", "TH"]
)


def analyse_rc(rc, config, out_dir, tname=""):
    all_distances = []
    for r in rc[:5]:
        r.load()
        this_out_dir = out_dir / r.get_name_for_save(config["allen_data_dir"])
        info = analyse_single_recording(
            r,
            config["gpfa_params"],
            this_out_dir,
            config["allen_data_dir"],
            ["CA1", "SUB", "TH"],
            filter_prop=config["allen_filter_properties"],
        )
        if info is None:
            continue

        distances_and_variances, scaled = perform_scaling_and_stats(
            info["correct"], info["incorrect"]
        )
        distances_and_variances["name"] = info["name"]
        all_distances.append(distances_and_variances)

        # Do plots
        n = r.get_name_for_save(config["allen_data_dir"])
        for name, data in scaled.items():
            fig = simple_trajectory_plot(data[0], data[1])
            fig = SimuranFigure(fig, out_dir / n / f"simple_trajectory_plot_{name}")
            fig.save()

            fig = plot_trajectories_split(data[0], data[1])
            fig = SimuranFigure(fig, out_dir / n / f"plot_trajectories_split_{name}")
            fig.save()
        r.unload()

    distances_df = pd.DataFrame(all_distances)
    distances_df.to_csv(out_dir / f"{tname}_distances.csv")


analyse_rc(rc, config, out_dir, "normal")

config["gpfa_params"]["decimate"] = False
out_dir = Path(
    r"E:\Repos\task-related-neural-activity\workflow\figures\scaling_no_decimate"
)
analyse_rc(rc, config, out_dir, "no_decimate")

config["gpfa_params"]["decimate"] = True
config["gpfa_params"]["gpfa_window"] = 20
out_dir = Path(r"E:\Repos\task-related-neural-activity\workflow\figures\scaling_20ms")
analyse_rc(rc, config, out_dir, "20ms")
